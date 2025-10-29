"""Simple rule-based data classifier for LLM training data.

Classifies texts into multiple tags or returns "Good".
"""

import warnings
import re
import logging

import fasttext
from huggingface_hub import hf_hub_download
import nltk
import spacy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

nltk.download("punkt", quiet=True)


def get_device():
    return 0 if torch.cuda.is_available() else -1


# Common flags

classification = "text-classification"

_unsafe_pipeline = None
_bias_pipeline = None
_toxicity_pipeline = None
_spam_pipeline = None
_advertisement_pipeline = None
_spacy_nlp = None
_quality_model = None
_known_info_tokenizer = None
_known_info_model = None

def load_unsafe():
    global _unsafe_pipeline
    if _unsafe_pipeline is None:
        print("Loading unsafe-pipeline...")
        _unsafe_pipeline = pipeline(
            model="meta-llama/Llama-Guard-3-1B",
            device=get_device(), task="text-generation"
        )

def unload_unsafe():
    global _unsafe_pipeline
    if _unsafe_pipeline is not None:
        del _unsafe_pipeline
        _unsafe_pipeline = None
        torch.cuda.empty_cache()

def load_bias():
    global _bias_pipeline
    if _bias_pipeline is None:
        print("Loading bias-pipeline...")
        _bias_pipeline = pipeline(
            model="valurank/distilroberta-bias",
            device=get_device(), task=classification
        )

def unload_bias():
    global _bias_pipeline
    if _bias_pipeline is not None:
        del _bias_pipeline
        _bias_pipeline = None
        torch.cuda.empty_cache()

def load_toxicity():
    global _toxicity_pipeline
    if _toxicity_pipeline is None:
        print("Loading toxicity-pipeline...")
        _toxicity_pipeline = pipeline(
            model="unitary/toxic-bert",
            device=get_device(), task=classification
        )

def unload_toxicity():
    global _toxicity_pipeline
    if _toxicity_pipeline is not None:
        del _toxicity_pipeline
        _toxicity_pipeline = None
        torch.cuda.empty_cache()

def load_spam():
    global _spam_pipeline
    if _spam_pipeline is None:
        print("Loading spam-pipeline...")
        _spam_pipeline = pipeline(
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            device=get_device(), task=classification,
        )

def unload_spam():
    global _spam_pipeline
    if _spam_pipeline is not None:
        del _spam_pipeline
        _spam_pipeline = None
        torch.cuda.empty_cache()

def load_advertisement():
    global _advertisement_pipeline
    if _advertisement_pipeline is None:
        print("Loading advertisement-pipeline...")
        _advertisement_pipeline = pipeline(
            model="0x7o/roberta-base-ad-detector",
            device=get_device(), task=classification
        )

def unload_advertisement():
    global _advertisement_pipeline
    if _advertisement_pipeline is not None:
        del _advertisement_pipeline
        _advertisement_pipeline = None
        torch.cuda.empty_cache()

def load_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("en_core_web_sm")

def load_quality():
    global _quality_model
    if _quality_model is None:
        print("Loading low-quality model...")
        _quality_model = fasttext.load_model(hf_hub_download(
            "kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2", "model_quantized.bin"
        ))

def load_known_info():
    global _known_info_tokenizer, _known_info_model
    if _known_info_model is None:
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

def unload_known_info():
    global _known_info_tokenizer, _known_info_model
    if _known_info_model is not None:
        del _known_info_tokenizer
        del _known_info_model
        _known_info_tokenizer = None
        _known_info_model = None
        torch.cuda.empty_cache()

replace_newlines = lambda text: re.sub("\n+", " ", text)

score_dict = {
    '__label__': 0, 
    '__label__Low': 0, 
    '__label__Mid': 1,
    '__label__High': 2
}


def is_unsafe(text: str) -> tuple[bool, float]:
    load_unsafe()
    results = _unsafe_pipeline(text)
    print("Unsafe results:", results)
    score = 1.0 if "unsafe" in results[0]["generated_text"].lower() else 0.0
    return (score > 0.0, score)


def is_toxic(text: str) -> tuple[bool, float]:
    load_toxicity()
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
    load_advertisement()
    result = _advertisement_pipeline(text)
    ad_scores = [res["score"] for res in result if res["label"] == "ad"]
    score = max(ad_scores) if ad_scores else 0.0
    return (score > 0.5, score)


def is_spammy(text: str) -> tuple[bool, float]:
    load_spam()
    result = _spam_pipeline(text)
    spam_scores = [res["score"] for res in result if res["label"] == "spam"]
    score = max(spam_scores) if spam_scores else 0.0
    return (score > 0.5, score)


def is_biased(text: str) -> tuple[bool, float]:
    load_bias()
    result = _bias_pipeline(text)
    bias_scores = [res["score"] for res in result if res["label"] == "biased"]
    score = max(bias_scores) if bias_scores else 0.0
    return (score > 0.6, score)


def has_sensitive_content(text: str) -> tuple[bool, float]:
    load_spacy()
    doc = _spacy_nlp(text)
    sensitive_entities = ["PERSON", "EMAIL", "PHONE", "GPE", "SSN"]
    has = any(ent.label_ in sensitive_entities for ent in doc.ents)
    score = 1.0 if has else 0.0
    return (has, score)


def unsafe_score(text: str) -> float:
    load_unsafe()
    results = _unsafe_pipeline(text)
    return 1.0 if "unsafe" in results[0]["generated_text"].lower() else 0.0

def toxic_score(text: str) -> float:
    load_toxicity()
    results = _toxicity_pipeline(text)
    toxic_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
    scores = [r["score"] for r in results if r["label"] in toxic_labels]
    return max(scores) if scores else 0.0

def scam_score(text: str) -> float:
    scam_patterns = [
        r"\b(buy|claim|win|free|offer|limited)\s+(now|today|urgent)",
        r"http[s]?://[^\s]+",
        r"\b(click|visit|call)\s+(here|below|now)",
    ]
    return 1.0 if any(re.search(pattern, text, re.IGNORECASE) for pattern in scam_patterns) else 0.0

def advertisement_score(text: str) -> float:
    load_advertisement()
    result = _advertisement_pipeline(text)
    ad_scores = [res["score"] for res in result if res["label"] == "ad"]
    return max(ad_scores) if ad_scores else 0.0

def spammy_score(text: str) -> float:
    load_spam()
    result = _spam_pipeline(text)
    spam_scores = [res["score"] for res in result if res["label"] == "spam"]
    return max(spam_scores) if spam_scores else 0.0

def biased_score(text: str) -> float:
    load_bias()
    result = _bias_pipeline(text)
    bias_scores = [res["score"] for res in result if res["label"] == "biased"]
    return max(bias_scores) if bias_scores else 0.0

def sensitive_content_score(text: str) -> float:
    load_spacy()
    doc = _spacy_nlp(text)
    sensitive_entities = ["PERSON", "EMAIL", "PHONE", "GPE", "SSN"]
    return 1.0 if any(ent.label_ in sensitive_entities for ent in doc.ents) else 0.0

def low_quality_score(text: str) -> float:
    load_quality()
    text_list = replace_newlines(text)
    pred = _quality_model.predict(text_list, k=-1)
    for l, s in zip(*pred):
        score = 0
        for _l, _s in zip(l, s):
            score += score_dict[_l] * _s
        return float(score)
    return 0.0

def known_information_score(text: str) -> float:
    load_known_info()
    inputs = _known_info_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(_known_info_model.device)
    with torch.no_grad():
        outputs = _known_info_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

def is_low_quality(text: str) -> tuple[bool, float]:
    load_quality()
    text_list = replace_newlines(text)
    pred = _quality_model.predict(text_list, k=-1)
    for l, s in zip(*pred):
        score = 0
        for _l, _s in zip(l, s):
            score += score_dict[_l] * _s
        score = float(score)
        return (score < 1.0, score)
    return (False, 0.0)


def is_known_information(text: str) -> tuple[bool, float]:
    load_known_info()
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
