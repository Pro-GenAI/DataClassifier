import json
from classifier import classify, classify_file
import tempfile
import os


def test_classify_high_quality():
    text = "This is a well-written example containing full sentences and clear meaning."
    assert classify(text) == "High Quality"


def test_classify_spammy_url():
    text = "Visit http://spam.example.com for great deals"
    assert classify(text) == "Spammy"


def test_classify_low_quality_short():
    text = "ok yes"
    assert classify(text) == "Low Quality"


def test_classify_low_quality_repetitive():
    text = "lol " * 10
    assert classify(text) == "Low Quality"


def test_classify_file_jsonl(tmp_path):
    data = [{"text": "Hello world"}, {"text": "Visit http://x"}, {"text": "lol lol lol"}]
    in_file = tmp_path / "in.jsonl"
    out_file = tmp_path / "out.jsonl"
    with open(in_file, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    classify_file(str(in_file), str(out_file), input_format="jsonl", text_field="text")
    labels = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels.append(obj.get("label"))

    # short "Hello world" is classified as Low Quality by the short-text rule
    assert labels == ["Low Quality", "Spammy", "Low Quality"]
