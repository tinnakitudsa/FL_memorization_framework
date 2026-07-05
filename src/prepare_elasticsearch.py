import torch
import os

import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

######## Elasticsearch #########
def create_elasticsearch(
    ELASTIC_PASSWORD,
    FINGERPRINT,
    IP="https://localhost:9200",
):
    es = Elasticsearch(
        IP,
        request_timeout=60,
        max_retries=100,
        retry_on_timeout=True,
        ssl_show_warn=False,
        basic_auth=("elastic", ELASTIC_PASSWORD),
        # verify_certs=False,
        ssl_assert_fingerprint=FINGERPRINT,
    )

    return es


def create_index(es, sentences_list, INDEX):

    # Create the index with mappings and settings
    es.indices.create(
        index=INDEX,
        body={
            "settings": {"number_of_shards": 20, "number_of_replicas": 0},
            "mappings": {"properties": {"text": {"type": "text"}}},
        },
    )

    # Verify index creation
    index_exists = es.indices.exists(index=INDEX)
    if index_exists:
        print(f"Index '{INDEX}' created successfully.")
    else:
        print(f"Failed to create index '{INDEX}'.")

    # Prepare the documents for bulk indexing
    actions = [
        {
            "_op_type": "index",  # Operation type
            "_index": INDEX,
            "_id": str(idx),
            "_source": {"text": sentence},
        }
        for idx, sentence in enumerate(sentences_list, start=1)
    ]

    # Bulk index the documents
    success, failed = bulk(es, actions)
    print(f"Successfully indexed {success} documents, {failed} failed.")

    es.indices.refresh(index=INDEX)

    # Verify number of documents indexed
    doc_count = es.count(index=INDEX)["count"]
    print(f"Number of documents indexed in '{INDEX}': {doc_count}")

    return es


def delete_all_index(es):
    # Enable deletion of all indices
    es.cluster.put_settings(
        body={"persistent": {"action.destructive_requires_name": False}}
    )

    # Delete all indices
    response = es.indices.delete(index="_all")
    print("Deleted all indices:", response)

    # Revert the setting for safety
    es.cluster.put_settings(
        body={"persistent": {"action.destructive_requires_name": True}}
    )

    return es


######### Sample Data from all Data##################
def save_eval_mem_dataset_dict(datasets_dict, base_path):
        
    for top_key, dataset in datasets_dict.items():
        
        data_name_path = os.path.join(base_path, top_key)
        dataset.save_to_disk(data_name_path)

def load_eval_mem_dataset_dict(base_path):
    output_dict = {}
    for folder in os.listdir(base_path):
        data_name_path = os.path.join(base_path, folder)
        output_dict[folder] = load_from_disk(data_name_path)  
    return output_dict

###########Preprocessing for Memorization Measurement########
def apply_chat_template(template, tokenizer, tokenized_prefix):

    chat_dict = {
        "qwen": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        "gpt2": "",
    }

    chat_token = tokenizer.encode(chat_dict[template], add_special_tokens=False)

    chat_tokenized_prefix = chat_token+tokenized_prefix

    return chat_tokenized_prefix

def chunk_list(big_list, batch_size):
    """
    Divides a big list into smaller chunks of specified batch size.

    Args:
        big_list (list): The list to be divided.
        batch_size (int): The size of each chunk.

    Returns:
        list of lists: A list containing the chunks.
    """
    return [big_list[i : i + batch_size] for i in range(0, len(big_list), batch_size)]

def process_data(template, dataset, tokenizer, prefix_length=50):   #input, abstract, Patient 

    splited_dict_prefix_suffix = []

    for record in tqdm(
        dataset,
        total=len(dataset),
        desc="Create Prefix and Suffix",
    ):

        tokens = tokenizer.encode(record["input"], add_special_tokens=False)
        tokenized_prefix = tokens[:prefix_length]
        tokenized_suffix = tokens[prefix_length:]

        prefix = tokenizer.decode(tokenized_prefix, skip_special_tokens=True)
        suffix = tokenizer.decode(tokenized_suffix, skip_special_tokens=True)


        tokenized_prefix=apply_chat_template(template, tokenizer, tokenized_prefix)
        # message = [{'content': prefix, 'role': 'user'}]
        # tokenized_prefix = tokenizer.apply_chat_template(message, tokenize=True, continue_final_message=True)



        splited_dict_prefix_suffix.append(
            {
                "prefix": prefix,
                "suffix": suffix,
                "tokenized_prefix": tokenized_prefix, #tokenized_prefix,
            }
        )

    return splited_dict_prefix_suffix


def get_decode_parameters(decode_method, decode_number):
    if decode_method == "greedy":
        decoding_params = {
            "num_beams": 1,  # DO Beam search
            # "temperature": 0.0,  # No temperature
            # "top_k": 0,  # No top-k sampling
            # "top_p": 1.0,  # No top-p sampling
            "do_sample": False,  # Disable sampling for greedy decoding
        }
    elif decode_method == "beamsearch":
        decoding_params = {
            "num_beams": decode_number,  # DO Beam search
            # "temperature": 0.0,  # No temperature
            # "top_k": 0,  # No top-k sampling
            # "top_p": 1.0,  # No top-p sampling
            "do_sample": False,  # Disable sampling for greedy decoding
        }
    elif decode_method == "temperature":
        decoding_params = {
            # "num_beams": 1,  # No beam search when sampling
            "temperature": decode_number,  # DO temperature
            # "top_k": 0,  # No top-k sampling
            # "top_p": 1.0,  # No top-p sampling
            "do_sample": True,  # Enable sampling
        }
    elif decode_method == "top_k":
        decoding_params = {
            # "num_beams": 1,  # No beam search when sampling
            # "temperature": 1.0,  # Default temperature
            "top_k": decode_number,  # DO top-k sampling
            # "top_p": 1.0,  # No top-p sampling
            "do_sample": True,  # Enable sampling
        }
    elif decode_method == "top_p":
        decoding_params = {
            # "num_beams": 1,  # No beam search when sampling
            # "temperature": 1.0,  # Default temperature
            # "top_k": 0,  # No top-k sampling
            "top_p": decode_number,  # DO top-p sampling
            "do_sample": True,  # Enable sampling
        }
    elif decode_method == "real_temperature":
        decoding_params = {
            # "num_beams": 1,  # No beam search when sampling
            "temperature": decode_number,  # DO temperature
            "top_k": 0,  # No top-k sampling
            # "top_p": 1.0,  # No top-p sampling
            "do_sample": True,  # Enable sampling
        }
    elif decode_method == "real_top_p":
        decoding_params = {
            # "num_beams": 1,  # No beam search when sampling
            # "temperature": 0.0,  # Default temperature
            "top_k": 0,  # No top-k sampling
            "top_p": decode_number,  # DO top-p sampling
            "do_sample": True,  # Enable sampling
        }
    else:
        raise "Use cirrect decode method: beamsearch, temperature, top_k, top_p"
    return decoding_params


def generate_suffixes(
    splited_dict_prefix_suffix,
    tokenizer,
    model,
    max_generated_length,
    decode_method,
    decode_number,
    batch_size=16,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = model.to(device)
    model.eval()

    splited_dict_prefix_suffix_chunks = chunk_list(
        splited_dict_prefix_suffix, batch_size
    )

    generated_suffixes_processed = []

    # Loop over the tokenized_prefix_chunks and prefix_list_chunks
    for tokenized_prefix_prefix_suffix in tqdm(

        splited_dict_prefix_suffix_chunks,
        total=len(splited_dict_prefix_suffix_chunks),
        desc="Generate Suffix",
    ):

        tokenized_prefix = [
            item["tokenized_prefix"] for item in tokenized_prefix_prefix_suffix
        ]
        
        tokenized_prefix = torch.tensor(tokenized_prefix).to(device)

        # Generate output sequences
        decoding_params = get_decode_parameters(decode_method, decode_number)
        outputs = model.generate(
            tokenized_prefix,
            max_length=max_generated_length,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,  #sep_token_id
            **decoding_params,
        )

        # Clean up any invalid token values
        # outputs[outputs < 0] = tokenizer.pad_token_id
        actual_output = [sample[len(prefix_token):] for prefix_token, sample in zip(tokenized_prefix, outputs)]
        # print("tokenized_prefix0",tokenized_prefix[0])
        # print("outputs", outputs[0])
        # print("sample[len(prefix):]", outputs[0][len(tokenized_prefix[0]):])
        # exit()
        # actual_False_generated_texts = tokenizer.batch_decode(actual_output, skip_special_tokens=False)
        generated_texts = tokenizer.batch_decode(actual_output, skip_special_tokens=True)
        # suffix_all = [
        #     item["suffix"] for item in tokenized_prefix_prefix_suffix
        # ]
        # prefix_all = [
        #     item["prefix"] for item in tokenized_prefix_prefix_suffix
        # ]
        # for fff, ttt, ppp, sss in  zip(actual_False_generated_texts, generated_texts, prefix_all, suffix_all):
        #     print(fff, "#######", ttt, "#######", ppp, "#######", sss)
        # exit()
        # generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Now process the generated text on CPU after the loop
        for idx, generated_suffix in enumerate(generated_texts):
            prefix = tokenized_prefix_prefix_suffix[idx]["prefix"]
            suffix = tokenized_prefix_prefix_suffix[idx]["suffix"]
            # if generated_text.startswith(prefix):
            #     generated_suffix = generated_text[len(prefix) :].strip()
            # else:
            #     generated_suffix = generated_text
            generated_suffixes_processed.append(
                {
                    "prefix": prefix,
                    "suffix": suffix,
                    "generated_suffix": generated_suffix,
                }
            )


    return generated_suffixes_processed
