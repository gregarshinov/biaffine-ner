from typing import Dict, List
import re
from functools import partial
from pathlib import Path
import json
import sys

import spacy
from spacy.tokens import Span, Doc
from datasets import load_dataset



def parse_raw_entities(doc: Doc, string_list: List[str]) -> List[Span]:
    '''
    This function takes in a Spacy Doc and a list of entities in NEREL format.
    After processing, it outputs a list of annotated Spacy Span objects, that are linked to the given Doc
    '''
    entities = []
    for raw_entity in string_list:
        ent_id, label_span, token_type = raw_entity.split("\t")
        if ";" not in label_span:
            label, start, end = label_span.split()
        else:
            match_obj = re.search("[A-Z_]+", label_span)
            label = match_obj[0]
            span_strings = label_span[match_obj.span()[1] + 1:].split(";")
            spans = [[int(number) for number in span.split()] for span in span_strings]
            for span in spans: # in this loop we split multi-span entities, flatten them so to say
                doc.char_span(int(span[0]), int(span[1]), label=label, alignment_mode='expand')
            continue
        ent = doc.char_span(int(start), int(end), label=label, alignment_mode='expand')
        entities.append(ent)
    return entities

def process_example(spacy_pipeline, example: Dict) -> Dict:
    '''
    example fields: ['id', 'text', 'entities', 'relations', 'links']
    returned fields: ['doc_key', 'ners', 'sentences']
    '''
    result = {"id": example["id"], "doc_key": example["id"], "ners": [], "sentences": []}
    text = example["text"]
    doc = spacy_pipeline(text)
    entities = parse_raw_entities(doc, example["entities"])
    for idx, sent in enumerate(doc.sents):
        result["ners"].append([])
        result["sentences"].append([str(tok) for tok in sent])
        for ent in entities:
            if ent.start >= sent.start and ent.end <= sent.end:
                start = str(ent.start - sent.start)
                end = str(ent.end - sent.start)
                label = doc.vocab.strings[ent.label]
                result["ners"][idx].append([start, end, label])
    return result

def dump_ds(ds, path: Path):
    with path.open(mode='w', encoding='utf-8') as f:
        for example in ds:
            record = example
            record["ners"] = [[[int(ner[0]), int(ner[1]), ner[2]] for ner in sent] for sent in example["ners"]]
            del record["id"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(prefix):
    nlp = spacy.load('ru_core_news_lg')
    ds = load_dataset("iluvvatar/NEREL")
    for split in ["train", "dev", "test"]:
        processed_ds = ds[split].map(partial(process_example, nlp), remove_columns=["text", "relations", "links"])
        path = Path(prefix) / f"{split}.jsonl"
        dump_ds(processed_ds, path)


if __name__ == "__main__":
    main(sys.argv[1])