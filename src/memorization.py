import os
import json
import torch
import random
import yaml
import pickle

import numpy as np
from datetime import datetime, timezone

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2TokenizerFast,
    AutoModelForCausalLM,
)
from datasets import load_from_disk, Dataset

from PAN2014 import read_parameters
from prepare_elasticsearch import (
    create_elasticsearch,
    delete_all_index,
    save_eval_mem_dataset_dict,
    load_eval_mem_dataset_dict,
)
from federated_pipeline import federated_memorization_pipeline

if torch.cuda.is_available():
    backend = "nccl"
else:
    backend = "gloo"



def unique_data(dataset, col):
    # Step 1: get all contexts
    contexts = dataset[col]

    # Step 2: build a set of first occurrence indices
    seen = {}
    first_indices = []
    for idx, c in enumerate(contexts):
        if c not in seen:
            seen[c] = True
            first_indices.append(idx)

    # Step 3: select rows by indices
    dataset = dataset.select(first_indices)

    return dataset

import argparse



parser = argparse.ArgumentParser(description="Argument parser for training script")

# parser.add_argument("--pretrained_model_name", type=str, required=True, help="Name of the pretrained model")
# parser.add_argument("--train_rule", type=str, required=True, help="Training rule to use")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
# parser.add_argument("--task_name", type=str, required=True, help="Task name")
parser.add_argument("--ft_data_paths", nargs="+", type=str, required=True, help="Path to fine-tuning data")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints")
parser.add_argument("--finetuned_model", type=str, required=True, help="Path to save/load finetuned model")
# parser.add_argument("--model_bit_type", type=str, required=True, help="Model bit type")
# parser.add_argument("--steps_list", default=None, help="List of steps")
parser.add_argument("--IP", type=str, help="IP address")
parser.add_argument("--elastic_password", type=str, help="ElasticSearch password")
parser.add_argument("--fingerprint", type=str, help="Unique fingerprint")
parser.add_argument("--parameters", type=str, help="Model parameters")
parser.add_argument("--plagiarism_model_path", type=str, help="Plagiarism detection model")
parser.add_argument("--prefix_length", type=int, default=0, help="Prefix length for training")
# parser.add_argument("--client_number", type=int, help="Client number")
# parser.add_argument("--eval_round", type=int, default=0, help="Evaluation round")
# parser.add_argument("--real_cols", type=str, default=None, help="real columns")
# parser.add_argument("--sampling_method_seed", type=int, default=5, help="Seed for sampling method")
parser.add_argument("--sampling_number_per_client", type=int, default=None, help="Number of samples per client")
# parser.add_argument("--stop_gen_id", type=int, default=None, help="stop_gen_id")
parser.add_argument("--decode_method", type=str, default="greedy", help="Decoding method")
parser.add_argument("--decode_number", default=None, help="Number of decoding iterations")
parser.add_argument("--max_generated_length", type=int, default=2048, help="Maximum length of generated text")
parser.add_argument("--template", type=str, default=None, help="Model Template")
parser.add_argument("--batch_size", type=str, default=None, help="Batch size")

args = parser.parse_args()

# task
# pretrained_model_name = args.pretrained_model_name
# train_rule = args.train_rule

# Set seed
seed = int(args.seed)

# data
# task_name = args.task_name
ft_data_paths = args.ft_data_paths
output_dir = args.output_dir

# model
finetuned_path = args.finetuned_model
# model_bit_type = args.model_bit_type
# steps_list = eval(args.steps_list) if args.steps_list else None

# Elasticsearch
IP = args.IP
ELASTIC_PASSWORD = args.elastic_password
FINGERPRINT = args.fingerprint

# PAN2014
PARAMETERS = args.parameters
PLAGIARISM_MODEL = args.plagiarism_model_path

###### Memorization Config ######
prefix_length = args.prefix_length
# client_number = args.client_number
template = args.template

# sampling
# eval_round = args.eval_round
# real_cols = args.real_cols
# sampling_method_seed = args.sampling_method_seed
sampling_number_per_client = args.sampling_number_per_client

# decode method
# stop_gen_id = args.stop_gen_id
decode_method = args.decode_method
decode_number = eval(args.decode_number)
max_generated_length = args.max_generated_length

batch_size = args.batch_size

# print("pretrained_model_name", pretrained_model_name)
# print("train_rule", train_rule)
print("seed", seed)
print("Loading Data")
print("IP", IP)
# print("task_name", task_name)
print("ft_data_paths", ft_data_paths)
print("output_dir", output_dir)
print("finetuned_path", finetuned_path)
# print("model_bit_type", model_bit_type)
# print("steps_list", steps_list)
print("PARAMETERS", PARAMETERS)
print("PLAGIARISM_MODEL", PLAGIARISM_MODEL)
print("prefix_length", prefix_length)
# print("client_number", client_number)
print("template", template)
# print("eval_round", eval_round)
# print("real_cols", real_cols)
# print("sampling_method_seed", sampling_method_seed)
print("sampling_number_per_client", sampling_number_per_client)
# print("stop_gen_id", stop_gen_id)
print("decode_method", decode_method)
print("decode_number", decode_number)
print("max_generated_length", max_generated_length)
print("batch_size", batch_size)

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)  # If you're using GPU
torch.cuda.manual_seed_all(seed)  # If you're using multiple GPUs

# running
print("Running")
print("IP", IP)
es = create_elasticsearch(
    ELASTIC_PASSWORD=ELASTIC_PASSWORD,
    FINGERPRINT=FINGERPRINT,
    IP=IP,
)
print(es.indices.exists(index="hello"))


# model_bit_type_dict={
#     "fp16": torch.float16,
#     "bf16": torch.bfloat16,
# }
# model_bit=model_bit_type_dict[model_bit_type]

pretrained_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
print("Tokenizer")
print(pretrained_tokenizer)
# print(
#     "Tokenizer is GPT2TokenizerFast:",
#     isinstance(pretrained_tokenizer, GPT2TokenizerFast),
# )
# if isinstance(pretrained_tokenizer, GPT2TokenizerFast):
#     pretrained_tokenizer = TokenizerType.from_pretrained(
#         tokenizer_path, model_max_length=1024
#     )
#     pretrained_tokenizer.padding_side = "right"
#     print("tokenizer.pad_token_id", pretrained_tokenizer.pad_token_id)
#     print("tokenizer.eos_token_id", pretrained_tokenizer.eos_token_id)
#     print("tokenizer.sep_token_id", pretrained_tokenizer.sep_token_id)

parameters = read_parameters(PARAMETERS)

# memorization evaluation tools
mem_tokenizer = AutoTokenizer.from_pretrained(PLAGIARISM_MODEL)
mem_model = AutoModelForSequenceClassification.from_pretrained(PLAGIARISM_MODEL)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Device:", device)
# placeholder_model = AutoModelForCausalLM.from_pretrained(
#         "/project/lt200252-wcbart/tinnakitu/mem_pFL/models/Qwen2.5-0.5B",
#         torch_dtype=torch.float16,
#         device_map=None,
#         trust_remote_code=True,
#         low_cpu_mem_usage=True,
#     ) 
# placeholder_model.to(device)
# save path

########Sampling Data################

saving_dir = os.path.join(
    output_dir,
    str(seed),
)
os.makedirs(saving_dir, exist_ok=True)

# path to save memorization evaluation data
original_federated_path = os.path.join(saving_dir, "original")


# check if there is federated version


if os.path.exists(original_federated_path):
    print("There is existing Data at", original_federated_path)

    memorization_train_partition = load_eval_mem_dataset_dict(original_federated_path)


else:

    print("New Method")
    
    print("Creating Mem Data at", original_federated_path)


    # create data
    train_dataset_partition = {}

    for filename in ft_data_paths:
        with open(filename, "r", encoding="utf-8") as f:
            jsonl_data = [json.loads(line) for line in f]
        train_dataset_partition[filename.split("/")[-1].replace(".jsonl", "")] = Dataset.from_list(jsonl_data)
    print(train_dataset_partition)


    print("Unique")
    train_dataset_partition = {
        k: unique_data(eval_data, "input") 
        for k, eval_data in train_dataset_partition.items()
    }
    print(train_dataset_partition)

    cutoff=prefix_length+20

    train_dataset_partition = {
        k: eval_data.filter(lambda example: len(pretrained_tokenizer(example["input"]).input_ids) >= cutoff )
        for k, eval_data in train_dataset_partition.items()
    }
    print("Filtering short samples")
    print(train_dataset_partition)

    if sampling_number_per_client:
        memorization_train_partition = {
            k: eval_data.select(np.random.default_rng(seed=seed).choice(len(eval_data), size=sampling_number_per_client, replace=False))
            for k, eval_data in train_dataset_partition.items()
        }
    else:
        memorization_train_partition = train_dataset_partition
    
    print("Sampling")
    print(memorization_train_partition)

    os.makedirs(saving_dir, exist_ok=True)
    save_eval_mem_dataset_dict(memorization_train_partition, original_federated_path)


print("federated memorization data")
print(memorization_train_partition)

mem_data = memorization_train_partition
# print(mem_data[0][0])



########### Memorization Measurement #########




timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
print(timestamp)
idx = f"seed{seed}_memseed{timestamp}".lower()
# memorization
federated_memorization_pipeline(
    # train_rule,
    # sampling_method_seed,
    mem_data,
    # pretrained_model_name,
    finetuned_path,
    # model_bit,
    prefix_length,
    max_generated_length,
    pretrained_tokenizer,
    template,
    mem_tokenizer,
    mem_model,
    es,
    idx,
    saving_dir,
    parameters,
    # stop_gen_id,
    decode_method,
    decode_number,
    # real_cols,
    # eval_round,
    seed,
    batch_size,
)

es = delete_all_index(es)

print("Completed")
