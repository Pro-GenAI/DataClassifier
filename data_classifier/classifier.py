"""Simple rule-based data classifier for LLM training data.

Classifies texts into multiple tags or returns "Good".
"""

import warnings
import re

import fasttext
from huggingface_hub import hf_hub_download
import nltk
import spacy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)


def get_device():
    return 0 if torch.cuda.is_available() else -1


# Common flags

classification = "text-classification"

print("Loading unsafe-pipeline...")
_unsafe_pipeline = pipeline(
    # model="QuantFactory/Llama-Guard-3-1B-GGUF",
    model="meta-llama/Llama-Guard-3-1B",
    device=get_device(), task="text-generation"
)
print("Loading bias-pipeline...")
_bias_pipeline = pipeline(
    model="valurank/distilroberta-bias",
    device=get_device(), task=classification
)
print("Loading toxicity-pipeline...")
_toxicity_pipeline = pipeline(
    model="unitary/toxic-bert",
    device=get_device(), task=classification
)
print("Loading spam-pipeline...")
_spam_pipeline = pipeline(
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    device=get_device(), task=classification,
)
print("Loading advertisement-pipeline...")
_advertisement_pipeline = pipeline(
    model="0x7o/roberta-base-ad-detector",
    device=get_device(), task=classification
)
_spacy_nlp = spacy.load("en_core_web_sm")


def is_unsafe(text: str) -> tuple[bool, float]:
    results = _unsafe_pipeline(text)
    print("Unsafe results:", results)
    score = 1.0 if "unsafe" in results[0]["generated_text"].lower() else 0.0
    return (score > 0.0, score)


def is_toxic(text: str) -> tuple[bool, float]:
    results = _toxicity_pipeline(text)
    # Get the max score for toxicity-related labels
    toxic_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
    scores = [r["score"] for r in results if r["label"] in toxic_labels]
    score = max(scores) if scores else 0.0
    return (score > 0.5, score)


def is_scam(text: str) -> tuple[bool, float]:
    # Heuristic augmentation for scams (e.g., urgency, suspicious URLs)
    scam_patterns = [
        r"\b(buy|claim|win|free|offer|limited)\s+(now|today|urgent)",
        r"http[s]?://[^\s]+",  # Suspicious links
        r"\b(click|visit|call)\s+(here|below|now)",
    ]
    score = 1.0 if any(re.search(pattern, text, re.IGNORECASE) for pattern in scam_patterns) else 0.0
    return (score > 0.0, score)


def is_advertisement(text: str) -> tuple[bool, float]:
    result = _advertisement_pipeline(text)
    ad_scores = [res["score"] for res in result if res["label"] == "ad"]
    score = max(ad_scores) if ad_scores else 0.0
    return (score > 0.5, score)


def is_spammy(text: str) -> tuple[bool, float]:
    result = _spam_pipeline(text)
    spam_scores = [res["score"] for res in result if res["label"] == "spam"]
    score = max(spam_scores) if spam_scores else 0.0
    return (score > 0.5, score)


def is_biased(text: str) -> tuple[bool, float]:
    result = _bias_pipeline(text)
    bias_scores = [res["score"] for res in result if res["label"] == "biased"]
    score = max(bias_scores) if bias_scores else 0.0
    return (score > 0.6, score)


def has_sensitive_content(text: str) -> tuple[bool, float]:
    doc = _spacy_nlp(text)
    sensitive_entities = ["PERSON", "EMAIL", "PHONE", "GPE", "SSN"]
    has = any(ent.label_ in sensitive_entities for ent in doc.ents)
    score = 1.0 if has else 0.0
    return (has, score)


# To detect low-quality text

print("Loading low-quality model...")
_quality_model = fasttext.load_model(hf_hub_download(
    "kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2", "model_quantized.bin"
))

replace_newlines = lambda text: re.sub("\n+", " ", text)

score_dict = {
    '__label__': 0, 
    '__label__Low': 0, 
    '__label__Mid': 1,
    '__label__High': 2
}

def is_low_quality(text: str) -> tuple[bool, float]:
    text_list = replace_newlines(text)
    pred = _quality_model.predict(text_list, k=-1)
    for l, s in zip(*pred):
        score = 0
        for _l, _s in zip(l, s):
            score += score_dict[_l] * _s
        score = float(score)
        return (score < 1.0, score)
    return (False, 0.0)


# To detect known information

print("Loading known-information model...")
_known_info_model_name = "microsoft/Phi-4-mini-instruct"
_known_info_tokenizer = AutoTokenizer.from_pretrained(_known_info_model_name)
if _known_info_tokenizer.pad_token is None:
    _known_info_tokenizer.pad_token = _known_info_tokenizer.eos_token

_known_info_model = AutoModelForCausalLM.from_pretrained(
    _known_info_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def is_known_information(text: str) -> tuple[bool, float]:
    # Tokenize the input text
    inputs = _known_info_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # Adjust as needed for model limits
    ).to(_known_info_model.device)
    
    # Compute the loss (negative log likelihood)
    with torch.no_grad():
        outputs = _known_info_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    # Threshold for "known" information; lower perplexity indicates higher predictability (likely known)
    threshold = 10.0  # Empirical threshold; may require tuning based on dataset
    return (perplexity < threshold, perplexity)


# Code to use the above functions to classify text

def classify(text: str) -> dict:
    """Return a flag or "Good" if data has a high-quality."""
    labels = []
    scores = {}
    flagged, score = is_unsafe(text)
    scores["unsafe"] = score
    if flagged:
        labels.append("Unsafe")
    flagged, score = is_toxic(text)
    scores["toxic"] = score
    if flagged:
        labels.append("Toxic")
    flagged, score = is_scam(text)
    scores["scam"] = score
    if flagged:
        labels.append("Scam")
    flagged, score = is_advertisement(text)
    scores["advertisement"] = score
    if flagged:
        labels.append("Advertisement")
    flagged, score = is_spammy(text)
    scores["spammy"] = score
    if flagged:
        labels.append("Spammy")
    flagged, score = is_biased(text)
    scores["biased"] = score
    if flagged:
        labels.append("Biased")
    flagged, score = has_sensitive_content(text)
    scores["sensitive_content"] = score
    if flagged:
        labels.append("Sensitive Content")
    flagged, score = is_low_quality(text)
    scores["low_quality"] = score
    if flagged:
        labels.append("Low Quality")
    flagged, score = is_known_information(text)
    scores["known_information"] = score
    if flagged:
        labels.append("Known Information")

    return {"labels": labels, "scores": scores}


if __name__ == "__main__":
    # quick local smoke test
    examples = [
        "This is a high quality example with complete sentences and clear meaning.",
        "Buy now http://spam.example.com!!!",
        "lol lol lol lol lol",
        "12345 67890 2345",
        "<html>some markup</html>",
    ]
    for e in examples:
        print(repr(e[:80]), "->", classify(e))
