"""Command-line wrapper around data_classifier.classifier

Usage examples:
  python scripts/classify.py --input examples/sample.jsonl --output out.jsonl --format jsonl
  python scripts/classify.py --text "Some text to classify"
"""
import argparse
import csv
import json
from pathlib import Path
import sys

from data_classifier.classifier import classify


def classify_file(input_path: str, output_path: str, input_format: str = "jsonl", text_field: str = "text"):
    """Classify a file of texts and write results.

    Supported input formats: 'jsonl', 'csv'.
    Output will be JSONL with added 'label' field.
    """
    input_format = input_format.lower()
    if input_format not in ("jsonl", "csv"):
        raise ValueError("input_format must be 'jsonl' or 'csv'")

    out_f = open(output_path, "w", encoding="utf-8")
    try:
        if input_format == "jsonl":
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(text_field, "")
                    obj["label"] = classify(text)
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            # csv
            with open(input_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get(text_field, "")
                    row["label"] = classify(text)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        out_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="Path to input file (jsonl or csv)")
    p.add_argument("--output", help="Path to output JSONL file when input is provided")
    p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Input file format")
    p.add_argument("--text", help="Classify a single text string and print the label")
    p.add_argument("--field", default="text", help="Field name containing text in input file (default: text)")
    args = p.parse_args()

    if args.text:
        print(classify(args.text))
        return

    if not args.input or not args.output:
        print("Either --text or both --input and --output must be provided", file=sys.stderr)
        p.print_usage()
        sys.exit(2)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file {input_path} not found", file=sys.stderr)
        sys.exit(2)

    classify_file(str(input_path), args.output, input_format=args.format, text_field=args.field)
    print(f"Wrote classified output to {args.output}")


if __name__ == "__main__":
    main()
