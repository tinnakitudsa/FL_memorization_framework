# Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning

[![ACL Anthology](https://img.shields.io/badge/ACL%20Anthology-2026.acl--short.56-4b5cc4)](https://aclanthology.org/2026.acl-short.56/)

## Git clone

```
git clone https://github.com/tinnakitudsa/FL_memorization_framework.git
cd FL_memorization_framework
```

## Environment

Python 3.10.18

## Setup

```
pip install -r requirement.txt --no-build-isolation
```

Then

```
python3 -m spacy download en_core_web_sm
python3 <<EOF
import nltk
nltk.download("stopwords")
nltk.download("punkt_tab")
EOF
```

You may need to change the name of punkt_tab if you get an error saying the file cannot be found.

### Elasticsearch

Dowload elasticsearch from https://www.elastic.co/downloads/past-releases/elasticsearch-8-16-1 and unzip it.

For Linux

```
curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.16.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.16.1-linux-x86_64.tar.gz
```

Try to run elasticsearch to get `ELASTIC_PASSWORD` and `FINGERPRINT`

```
PORT_NAME=53000
$ELASTICSEARCH/bin/elasticsearch -E http.port=$PORT_NAME > elasticsearch.log 2>&1 &
```

When url is up,

`ELASTIC_PASSWORD` set password

```
$ELASTICSEARCH/bin/elasticsearch-reset-password -u elastic -i --url "https://localhost:$PORT_NAME"
```

`FINGERPRINT` get fingerprint

```
openssl x509 -fingerprint -sha256 -noout -in $ELASTICSEARCH/config/certs/http_ca.crt | cut -d'=' -f2 | tr -d ':'
```

### Plagiarism Detection Model

Dowload https://huggingface.co/Intel/roberta-base-mrpc

### Environment Variables

```
ELASTICSEARCH=<your own path>/elasticsearch-8.16.1
ELASTIC_PASSWORD=<your own password>
FINGERPRINT=<your own fingerprint>

PLAGIARISM_MODEL=<your own path>/roberta-base-MRPC
```

## Data Format

JSONL format

```
{"input": "", "output": "", "instruction": ""}
{"input": "", "output": "", "instruction": ""}
...
```

Only text in `input` will be measured by this memorization. Other fields are not neccesary.

## Run

```
bash eval.sub
```
