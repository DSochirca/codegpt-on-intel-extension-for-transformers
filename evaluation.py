import gc
import os
import shutil
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from intel_extension_for_transformers.optimization.pipeline import pipeline
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=True, nb_workers=16)
tqdm.pandas()

from fuzzywuzzy import fuzz
import re
from memory_profiler import memory_usage

# Load the dataset and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py", truncation_side='left')
dataset = load_dataset("0n1xus/codexglue", 'code-completion-token-py150')
test = dataset['test'].to_pandas()


# We split the input code snippet and take the last line as the expected output and the rest as input
def split_input_output(x):
    split_seq = x['code'].split('<EOL>')
    return ['<EOL>'.join(split_seq[0:-1]) + '<EOL>', split_seq[-1]]


test[['input', 'output']] = test.progress_apply(split_input_output, axis=1, result_type="expand")

# We only use 1000 samples for evaluation
test = test.sample(1000, random_state=42)


def eval_model(model, model_name, path):
    """
    Evaluate a model on the test set
    :param model: the model to evaluate
    :param model_name: the name of the model
    :param path: the path of the (zipped) model to measure its disk size
    :return: a dictionary of evaluation results
    """
    # Create a pipeline to generate with the model
    generator = pipeline('text-generation',  # Type of pipeline
                         model=model,
                         tokenizer=tokenizer,
                         framework="pt",
                         device=torch.device('cpu'))

    def generate(sample):
        """
        Get the model's generated output for a sample input
        """
        output_tokens = tokenizer(sample['output'], truncation=True, max_length=1024)
        num_tokens = 1024 - (len(output_tokens) + 2)
        truncated_input = tokenizer.decode(tokenizer.encode(sample['input'])[-num_tokens:])
        return generator(truncated_input, max_new_tokens=len(output_tokens))[0]['generated_text']

    def apply_function():
        """
        Run generation and measure memory usage. We need to run it sequentially, otherwise the memory usage will impact the inference speed.
        """
        test['gen_output'] = test.progress_apply(generate, axis=1)

    mem_usage = memory_usage(apply_function)

    # Shamelessly stolen from:
    # https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-line/evaluator/evaluator.py
    def post_process(code):
        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
        return code

    def score(batch):
        """
        Calculate the edit similarity and exact match metrics
        """
        pred = post_process(batch['gen_output'].strip().replace('<s>', '').split('<EOL>')[-1])
        gt = post_process(batch['output'].strip().replace('</s>', ''))
        edit_sim = fuzz.ratio(pred, gt)
        EM = 0
        if pred.split() == gt.split():
            EM = 1
        return edit_sim, EM

    # Measure the disk size of the zipped model directory
    disk_size = os.path.getsize(path) / 1024 / 1024
    disk_size = disk_size * 1.04858  # use MB instead of MiB

    # Measure ES & EM
    test[['edit_sim', 'EM']] = test.progress_apply(score, axis=1, result_type="expand")

    def inference_time(sample):
        """
        Measure the inference time of a sample
        """
        start_time = time.time()
        generate(sample)
        end_time = time.time()
        return end_time - start_time

    # Measure the average CPU inference time
    num_samples = len(test)
    cpu_inference = num_samples / test.progress_apply(inference_time, axis=1).sum()

    return {
        'Model name': model_name,
        'Disk size': disk_size,
        'Mem usage': max(mem_usage),
        'CPU inf': cpu_inference,
        'Edit sim': sum(list(test['edit_sim'])) / len(test),
        'EM': sum(list(test['EM'])) / len(test) * 100
    }


def get_eval_results(model_names, model_paths):
    """
    Evaluate a list of models
    :param model_names: the names of the models (to be displayed in a table)
    :param model_paths: the disk paths to the models
    :return: a dataframe of evaluation results
    """
    results = []
    for name, path in zip(model_names, model_paths):
        gc.collect()  # clear memory to minimize inference and memory usage noise in the results

        model = AutoModelForCausalLM.from_pretrained(path)

        # We save the model locally and zip it in order to measure its disk size
        path = './models/' + path.replace('AISE-TUDelft/', '')  # save the model in the "models" directory
        model.save_pretrained(path)

        # Zip the model directory:
        zip_dir = path.replace('models', 'models_zipped')  # "models_zipped" directory
        shutil.make_archive(zip_dir, 'zip', path)
        zip_dir = zip_dir + '.zip'

        results.append(eval_model(model, name, zip_dir))

        # clear model from memory to minimize inference and memory usage noise in the results
        del model
        gc.collect()

    results = pd.DataFrame(results)
    results = results[['Model name', 'Disk size', 'Mem usage', 'CPU inf', 'Edit sim', 'EM']].round(2)
    return results


# ----------------------------
# PRUNING

model_original = "AISE-TUDelft/CodeGPT-Py150"

models_pruned = ["AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.5-sparsity",
                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.4-sparsity",
                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.6-sparsity",
                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.7-sparsity",
                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.8-sparsity",
                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.9-sparsity"]

model_paths = [model_original] + models_pruned
names = ['baseline', 'sparsity_40', 'sparsity_50', 'sparsity_60', 'sparsity_70', 'sparsity_80', 'sparsity_90']

results_pruning = get_eval_results(names, model_paths)
print(results_pruning)

results_pruning.to_csv('./results/pruning.csv', index=False)
del results_pruning

# ----------------------------
# QUANTIZATION - ALL LAYERS

model_original = "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.6-sparsity"

models_quantized_all_layers = ["AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-channel",
                               "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-channel",
                               "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-asym-per-tensor",
                               "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-only-weights-sym-per-tensor"]

model_paths = [model_original] + models_quantized_all_layers
names = ['pruned_sparsity_60', 'asym_per_channel', 'sym_per_channel', 'asym_per_tensor', 'sym_per_tensor']

results_quant_all_layers = get_eval_results(names, model_paths)
print(results_quant_all_layers)

results_quant_all_layers.to_csv('./results/quant_all_layers.csv', index=False)
del results_quant_all_layers

# ----------------------------
# QUANTIZATION - ONLY WEIGHTS

model_original = "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-pruned-0.6-sparsity"
models_quantized_only_weights = ["AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-asym-per-channel",
                                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-sym-per-channel",
                                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-asym-per-tensor",
                                 "AISE-TUDelft/BRP-Sochirca-CodeGPT-Py150-0.6-sparse-q-all-layers-sym-per-tensor"]

model_paths = [model_original] + models_quantized_only_weights
names = ['pruned_sparsity_60', 'asym_per_channel', 'sym_per_channel', 'asym_per_tensor', 'sym_per_tensor']

results_quant_only_weights = get_eval_results(names, model_paths)
print(results_quant_only_weights)

results_quant_only_weights.to_csv('./results/quant_only_weights.csv', index=False)
