import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, EvalPrediction, set_seed

from intel_extension_for_transformers.optimization import metrics, PrunerConfig, PruningConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=8)

from tqdm import tqdm

tqdm.pandas()

from fuzzywuzzy import fuzz

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs so that the fine-tuning is done on CPU
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb logging when fine-tuning

# Load trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("AISE-TUDelft/CodeGPT-Py150")
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py", truncation_side='left')

# Load dataset from hub: https://huggingface.co/datasets/0n1xus/codexglue/viewer/code-completion-token-py150
dataset = load_dataset("0n1xus/codexglue", 'code-completion-token-py150')

# The entire dataset is too big for fine-tuning on CPUs, so we take a subset of 128 samples
# We also change dataset to pandas dataframe (there does not seem to be a way to apply a func with two outputs using HF datasets 😩)
train = dataset['train'].select(range(128)).to_pandas()
valid = dataset['validation'].select(range(128)).to_pandas()


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
    # We need at least 2 training/fine-tuning epochs in order to prune the model to the desired sparsity ratio
    num_train_epochs=2,
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


# After pruning the library evaluates the model, but we want to do that in a separate script,
# so we define an empty eval function to override the default one
def empty_eval_func(_model):
    return []


# Pruning function which takes a sparsity ratio as argument and prunes the model
def prune(sparsity_ratio):
    """
    Prune the model to the desired sparsity ratio using the Intel Extension for Transformers library
    :param sparsity_ratio: target sparsity ratio
    """

    set_seed(42)

    trainer = NLPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    metric = metrics.Metric(name="eval_loss")  # will be used to measure the performance of tuned models
    pruner_config = PrunerConfig(
        epoch_range=[0, 1],  # prune for one epoch
        prune_type="GroupLasso",  # pruning type, BasicMagnitude and PatternLock are also available
        target_sparsity_ratio=sparsity_ratio,  # target sparsity ratio
        # Some layers will not be pruned, because they are not compatible with the structured tile pattern 1x2
        names=[n for n, p in model.named_parameters() if
               ('bias' not in n and ('ln_f' not in n) and ('ln_1' not in n) and ('ln_2' not in n))],
        parameters={"alpha": 0.006,  # regularization parameter
                    "start_step": 1,
                    "end_step": 1,  # one shot pruning (pruning in one single step)
                    "pattern": "tile_pattern_1x2"},
        # pruning parameters in entire blocks (in this case 1 output channel and 2 input channels)
    )
    p_conf = PruningConfig(pruner_config=[pruner_config], metrics=metric)

    # Force pruning to happen in one step:
    p_conf.inc_config.usr_cfg.pruning.approach.weight_compression.start_epoch = 1
    p_conf.inc_config.usr_cfg.pruning.approach.weight_compression.end_epoch = 1

    # Prune the model
    model_pruned = trainer.prune(
        pruning_config=p_conf,
        eval_func=empty_eval_func,
        train_func=trainer.builtin_train_func,
    )
    trainer.model = model_pruned

    # Save the pruned model
    trainer.save_model(f'./models/CodeGPT-Py150-pruned-{round(sparsity_ratio, 1)}-sparsity')


# Define sparsity ratios to prune for
sparsity_ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Prune for each sparsity ratio:
for i in range(len(sparsity_ratios)):
    sr = sparsity_ratios[i]
    prune(sr)
