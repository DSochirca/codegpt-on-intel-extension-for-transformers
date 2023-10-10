import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, EvalPrediction, set_seed
import pandas as pd

from intel_extension_for_transformers.optimization import metrics, objectives, QuantizationConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=8)

from tqdm import tqdm

tqdm.pandas()
from fuzzywuzzy import fuzz

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs so that the fine-tuning is done on CPU
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb logging when fine-tuning

# Load trained model and tokenizer:
model = AutoModelForCausalLM.from_pretrained("AISE-TUDelft/CodeGPT-Py150-pruned-0.6-sparsity")
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py", truncation_side='left')

# Load dataset from hub: https://huggingface.co/datasets/0n1xus/codexglue/viewer/code-completion-token-py150
dataset = load_dataset("0n1xus/codexglue", 'code-completion-token-py150')

# The entire dataset is too big for quantizing on CPUs, so we take a subset of 100 samples
# We also change dataset to pandas dataframe (there does not seem to be a way to apply a func with two outputs using HF datasets 😩)
train = dataset['train'].select(range(100)).to_pandas()
valid = dataset['validation'].select(range(100)).to_pandas()


# We split the input code snippet and take the last line as the expected output and the rest as input
def split_input_output(x):
    split_seq = x['code'].split('<EOL>')
    return ['<EOL>'.join(split_seq[0:-1]) + '<EOL>', split_seq[-1]]


# Tokenize the sequences
# Note: CodeGPT supports 1024 tokens max so we truncate longer sequences
# We add padding to make sure all sequences the same length
def tokenize_both(batch):
    return tokenizer(batch['input'], batch['output'], padding="max_length", truncation=True, max_length=1024, return_tensors='pt')


# Apply split function to the Dataframes
train[['input', 'output']] = train.progress_apply(split_input_output, axis=1, result_type="expand")
valid[['input', 'output']] = valid.progress_apply(split_input_output, axis=1, result_type="expand")

# Tokenize train and valid
train_ds = Dataset.from_pandas(train, split="train").map(tokenize_both, num_proc=64)
valid_ds = Dataset.from_pandas(valid, split="validation").map(tokenize_both, num_proc=64)

# Training arguments
training_args = TrainingArguments(
    no_cuda=True,  # Run on CPU
    do_train=True,  # Perform training
    do_eval=True,  # Perform evaluation

    output_dir="./test_epoch",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=1,  # number of training epochs
    per_device_train_batch_size=1,  # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps=200,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    prediction_loss_only=True,  # consider only loss in evaluation
    seed=42,  # Seed for rng
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors='pt'
)


# Evaluation function
def compute_metrics(p: EvalPrediction):
    # Get predicted tokens
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    # Compute edit similarity score
    edit_sim_scores = [fuzz.ratio(pred, gt) for pred, gt in zip(preds, p.label_ids)]
    edit_sim = np.mean(edit_sim_scores)

    return {"accuracy": edit_sim}


# Quantization parameters
tune_metric = metrics.Metric(name="eval_samples_per_second")  # we want to maximize the number of samples per second
objective = objectives.performance  # 2 objectives are supported: model_size and performance (we want to maximize performance)
sampling_size = len(train_ds)  # number of samples to use for tuning


def quantize(op_names, model_name):
    """
    Quantize the model using the Intel Extension for Transformers library
    :param op_names: A dictionary of layer names and their quantization configuration
    :param model_name: The name of the model to be quantized (for exporting)
    """

    set_seed(42)
    trainer = NLPTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=data_collator,
                         tokenizer=tokenizer, compute_metrics=compute_metrics)

    quantization_config = QuantizationConfig(approach="PostTrainingDynamic",
                                             timeout=1000000,  # tuning timeout in seconds, set to an arbitrarily large value
                                             max_trials=9208512000,  # max number of tune times, set to an arbitrarily large value
                                             metrics=[tune_metric], objectives=[objective], sampling_size=sampling_size)

    quantization_config.op_wise = op_names  # specify the quantization configuration for each layer
    quantization_config.random_seed = 42

    # Quantize the model
    model_quantized = trainer.quantize(quantization_config, train_func=trainer.builtin_train_func, eval_func=trainer.builtin_eval_func)

    if model_quantized is not None:
        trainer.save_model(f'./models/{model_name}')


# -----------------------------
# Quantization (only weights, symmetric per channel)

op_name_dict = {}
layers = [n for n, p in model.named_parameters()]  # Get all layer names
for key in layers:
    op_name_dict[key] = {
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        },
        "activation": {
            "dtype": ["fp32"]  # We do not quantize activations
        }
    }

quantize(op_name_dict, 'CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-channel')

# -----------------------------
# Quantization (only weights, asymmetric per channel)

op_name_dict = {}
layers = [n for n, p in model.named_parameters()]
for key in layers:
    op_name_dict[key] = {
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["asym"]
        },
        "activation": {
            "dtype": ["fp32"]  # We do not quantize activations
        }
    }

quantize(op_name_dict, 'CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-channel')

# -----------------------------
# Quantization (only weights, asymmetric, per-tensor)

op_name_dict = {}
layers = [n for n, p in model.named_parameters()]
for key in layers:
    op_name_dict[key] = {
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_tensor"],
            "scheme": ["asym"]
        },
        "activation": {
            "dtype": ["fp32"]  # We do not quantize activations
        }
    }

quantize(op_name_dict, 'CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-tensor')

# -----------------------------
# Quantization (only weights, symmetric, per-tensor)

op_name_dict = {}
layers = [n for n, p in model.named_parameters()]
for key in layers:
    op_name_dict[key] = {
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "activation": {
            "dtype": ["fp32"]  # We do not quantize activations
        }
    }

quantize(op_name_dict, 'CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-tensor')
