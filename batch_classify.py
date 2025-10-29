"""Batch classify a dataset with progress tracking and model unloading.

Usage:
  python scripts/batch_classify.py --input data.jsonl --output labels.json
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from classifier import (
    unsafe_score, toxic_score, scam_score, advertisement_score, spammy_score, biased_score,
    sensitive_content_score, low_quality_score, known_information_score,
    load_unsafe, load_toxicity, load_bias, load_spam, load_advertisement,
    unload_unsafe, unload_toxicity, unload_bias, unload_spam, unload_advertisement,
    load_spacy, load_quality, load_known_info, unload_known_info
)

progress_file = "progress.json"
scores_file = "scores.json"
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {}

    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            scores = json.load(f)
    else:
        scores = {}

    return progress, scores

def save_progress(progress, scores):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)

def load_data_from_file(input_file):
    data = []
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_id = obj.get("id", str(i))
            text = obj.get("text", "")
            data.append({"id": row_id, "text": text, "obj": obj})
    return data

def load_data_from_hf(dataset_name, subset="default", split="train", num_samples=100):
    cache_filename = f".cache_{dataset_name.replace('/', '_')}_{subset}_{split}.json"
    data = []
    if os.path.exists(cache_filename):
        with open(cache_filename, "r") as f:
            data = json.load(f)
    if len(data) < num_samples:
        # Use num_samples to build the split slice and pass subset as the `name` argument when provided.
        dataset = load_dataset(dataset_name, name=subset,
                               split=split, streaming=True)
        dataset.skip(len(data)) # type: ignore
        for i, item in enumerate(dataset):
            if len(data) >= num_samples:
                break
            id = item.get("id", str(i))
            text = item.get("text", "")
            if not text:
                continue
            data.append({"id": id, "text": text, "obj": item})
        with open(cache_filename, "w") as f:
            json.dump(data, f, indent=2)
    return data

def main():
    parser = argparse.ArgumentParser(description="Batch classify dataset")
    parser.add_argument("--input", help="Input JSONL file")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb", help="HuggingFace dataset name if no input file")
    parser.add_argument("--subset", default="CC-MAIN-2025-26", help="Subset of the HuggingFace dataset to use")
    parser.add_argument("--split", default="train", help="Split of the HuggingFace dataset to use")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to load from HF dataset")
    args = parser.parse_args()

    if args.input:
        dataset_name = Path(args.input).stem
        data = load_data_from_file(args.input)
        print(f"Loaded {len(data)} rows from {args.input}")
    else:
        dataset_name = args.dataset
        data = load_data_from_hf(dataset_name, subset=args.subset, num_samples=args.num_samples)
        print(f"Loaded {len(data)} rows from HuggingFace dataset {dataset_name}")
    output_file = f"labels_{dataset_name}.json"


    classifiers = [
        {"name": "unsafe", "score_func": unsafe_score, "load": load_unsafe, "unload": unload_unsafe, "threshold": 0.0, "compare": ">"},
        {"name": "toxic", "score_func": toxic_score, "load": load_toxicity, "unload": unload_toxicity, "threshold": 0.5, "compare": ">"},
        {"name": "scam", "score_func": scam_score, "load": None, "unload": None, "threshold": 0.0, "compare": ">"},
        {"name": "advertisement", "score_func": advertisement_score, "load": load_advertisement, "unload": unload_advertisement, "threshold": 0.5, "compare": ">"},
        {"name": "spammy", "score_func": spammy_score, "load": load_spam, "unload": unload_spam, "threshold": 0.5, "compare": ">"},
        {"name": "biased", "score_func": biased_score, "load": load_bias, "unload": unload_bias, "threshold": 0.6, "compare": ">"},
        {"name": "sensitive_content", "score_func": sensitive_content_score, "load": load_spacy, "unload": None, "threshold": 0.0, "compare": ">"},
        {"name": "low_quality", "score_func": low_quality_score, "load": load_quality, "unload": None, "threshold": 1.0, "compare": "<"},
        {"name": "known_information", "score_func": known_information_score, "load": load_known_info, "unload": unload_known_info, "threshold": 10.0, "compare": "<"},
    ]

    progress, scores = load_progress()
    if not scores:
        scores = {row["id"]: {} for row in data}

    for classifier in classifiers:
        name = classifier["name"]
        print(f"\nProcessing classifier: {name}")
        if classifier["load"]:
            classifier["load"]()
        current_row = progress.get(name, 0)
        for i in range(current_row, len(data)):
            print(f"Classifying row {i+1}/{len(data)} for {name}", end="\r")
            row = data[i]
            score = classifier["score_func"](row["text"])
            scores[row["id"]][name] = score
            progress[name] = i + 1
            if (i + 1) % 100 == 0:
                save_progress(progress, scores)
                print(f"Processed {i+1}/{len(data)} for {name}")
        save_progress(progress, scores)
        if classifier["unload"]:
            classifier["unload"]()
        print(f"Completed {name}")

    # Compute labels
    labels_data = {}
    for row in data:
        row_id = row["id"]
        text = row["text"]
        scores_dict = scores[row_id]
        labels = []
        for classifier in classifiers:
            name = classifier["name"]
            score = scores_dict[name]
            threshold = classifier["threshold"]
            compare = classifier["compare"]
            if (compare == ">" and score > threshold) or (compare == "<" and score < threshold):
                labels.append(name)
        labels_data[row_id] = {
            "text": text,
            "labels": labels,
            "scores": scores_dict
        }

    output_data = {dataset_name: labels_data}
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved labels to {output_file}")

if __name__ == "__main__":
    main()