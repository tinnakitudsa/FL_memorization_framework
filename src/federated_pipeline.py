import os
import json
import torch
import transformers
from transformers import AutoModelForCausalLM
from PAN2014 import memorization_evaluate
from prepare_elasticsearch import (
    create_index,
    process_data,
    generate_suffixes,
)

# each models genereate data from prefix from all clients' datasets Then, test memorization.
def federated_memorization_pipeline(
    # train_rule,
    mem_datasets,
    # pretrained_model_name,
    finetuned_paths,
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
    decode_method,
    decode_number,
    # real_cols,
    # eval_round,
    seed,
    batch_size,
):

    client_list = list(mem_datasets.keys())
    # for each local model
    for client_name in client_list:

        print("######################Client", client_name, ":")
        print("Getting ground truth suffix")

        ######## Split prefix and suffix ########
        prefix_suffix_path_dir = os.path.join(saving_dir, "split_prefix_suffix")
        os.makedirs(prefix_suffix_path_dir, exist_ok=True)
        prefix_suffix_path = os.path.join(
            prefix_suffix_path_dir,
            f"fed_prefix_suffix_dict_seed{str(seed)}.jsonl",
        )

        if os.path.exists(prefix_suffix_path):
            print("There is existing data. Loading", prefix_suffix_path)
            with open(prefix_suffix_path, "r") as f:
                federated_splited_dict_prefix_suffix = json.load(f)

        else:
            print("Creating data", prefix_suffix_path)
            federated_splited_dict_prefix_suffix = get_federated_ground_truth_suffix(
                template, mem_datasets, pretrained_tokenizer, prefix_length, 
            )
            # save
            print("Saved:", prefix_suffix_path)
            with open(prefix_suffix_path, "w") as file:
                json.dump(federated_splited_dict_prefix_suffix, file)



        ######## Generate suffix ########
        generated_suffix_dir = os.path.join(
            saving_dir,
            "generated_suffix",
            f"{seed}",
            f"{decode_method}{decode_number}",
        )

        os.makedirs(generated_suffix_dir, exist_ok=True)
        generated_suffix_path = os.path.join(generated_suffix_dir, f"generated_suffix_dict_client{client_name}.jsonl")

        print("Getting generated suffix")
        if os.path.exists(generated_suffix_path):
            print("There is existing data. Loading")

            with open(generated_suffix_path, "r") as f:
                generated_suffixes = json.load(f)

        else:
            generated_suffixes = {}

            for (
                data_idx,
                splited_dict_prefix_suffix,
            ) in federated_splited_dict_prefix_suffix.items():
                print("Data", data_idx)
                # print("train_rule", train_rule)

                print("Generating suffix with model:", finetuned_paths)
                model_path = finetuned_paths

                # get model
                model = AutoModelForCausalLM.from_pretrained(
                    finetuned_paths,
                    # torch_dtype=model_bit,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ) 

                transformers.logging.set_verbosity_error()

                generated_suffixes_processed = generate_suffixes(
                    splited_dict_prefix_suffix,
                    pretrained_tokenizer,
                    model,
                    max_generated_length,
                    decode_method,
                    decode_number,
                    batch_size,
                )
                transformers.logging.set_verbosity_info()

                torch.cuda.synchronize()

                generated_suffixes[data_idx] = generated_suffixes_processed

            print("Saved:", generated_suffix_path)
            with open(generated_suffix_path, "w") as file:
                json.dump(generated_suffixes, file)

        ######## Memorization Measurement ########
        print("Evaluating")
        federated_memorization(
            # train_rule,
            generated_suffixes,
            mem_tokenizer,
            mem_model,
            es,
            idx,
            parameters,
            saving_dir,
        )


# get dict of all prefix/suffix from all clients
def get_federated_ground_truth_suffix(
    template, mem_datasets, pretrained_tokenizer, prefix_length,
):
    prefix_suffix_dict = {}
    for key, dataset in mem_datasets.items():
        print("Data", key)

        # tokenized_prefix, prefixs, suffixes
        splited_dict_prefix_suffix = process_data(
            template=template,
            dataset=dataset,
            tokenizer=pretrained_tokenizer,
            prefix_length=prefix_length,
            # real_cols=real_cols,
        )
        prefix_suffix_dict[key] = splited_dict_prefix_suffix

    return prefix_suffix_dict

# data from all clients vs one FT model
def federated_memorization(
    # train_rule,
    generated_suffixes,
    # suffixes,
    # prefixes,
    mem_tokenizer,
    mem_model,
    es,
    idx,
    parameters,
    saving_dir,
):
    # memorization

    # generated_suffix
    for key, generated_suffixes_processed in generated_suffixes.items():
        print("===========================Data", key)
        # count mem per data
        for target_key in generated_suffixes.keys():
            print("---------------Targeted Data", target_key)

            INDEX = f"{idx}_client{target_key.replace('.txt', '').lower()}_data"
            print(len(generated_suffixes[target_key]))
            # Don't create index if it exist
            print("INDEX", INDEX)
            try:
                exists = es.indices.exists(index=INDEX)
                print(f"Index exists: {exists}")
            except Exception as e:
                print(f"Error checking if index exists: {e}")
            if not es.indices.exists(index=INDEX):
                suffixes = [e["suffix"] for e in generated_suffixes[target_key]]
                es = create_index(es, suffixes, INDEX)


            index_len = es.count(index=INDEX)["count"]
            print("Suffixes in index", index_len)
            top_k_match = 10
            print("Match top", top_k_match)

            memorization_log = memorization_evaluate(
                es,
                mem_model,
                generated_suffixes_processed,
                mem_tokenizer,
                parameters,
                INDEX,
                top_k_match,
            )

            memorization_log_dir = os.path.join(
                saving_dir,
                f"memorization_log_prefix_{key.replace('.txt', '')}_suffix_{target_key.replace('.txt', '')}.json",
            )

            print("Saved:", saving_dir)
            with open(memorization_log_dir, "w") as file:
                json.dump(memorization_log, file)