# NER BERT Adapter

Adapter for [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER).

## Overview
Fine-tuned BERT model for Named Entity Recognition. Identifies PER, ORG, LOC, and MISC entities in general and legal text.

## Model details
- **Model:** dslim/bert-base-NER
- **Task:** extract
- **Domain:** general, legal
- **License:** MIT

## Output schema
The pipeline returns a list of entity dicts:
`json
[{"entity": "B-PER", "score": 0.998, "word": "John", "start": 0, "end": 4}]
`

## Usage
`python
from transformers import pipeline
from ner_bert_adapter import NERBertAdapter

pipe = pipeline("ner", model="dslim/bert-base-NER")
adapter = NERBertAdapter()
model_input = adapter.ingress(ir)
output = pipe(model_input["text"])
result_ir = adapter.egress(output, ir, latency_ms=42)
`

## License
MIT
