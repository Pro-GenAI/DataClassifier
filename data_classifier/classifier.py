"""Simple rule-based data classifier for LLM training data.

Classifies texts into multiple tags or returns "Good".
"""

import warnings
import re

import torch
from transformers import pipeline
import spacy
import nltk
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import fasttext

warnings.filterwarnings("ignore")

# Download NLTK data if needed
nltk.download("punkt", quiet=True)


def get_device():
    return 0 if torch.cuda.is_available() else -1


# Common flags

classification = "text-classification"
_unsafe_pipeline = pipeline(
    model="QuantFactory/Llama-Guard-3-1B-GGUF",
    device=get_device(), task=classification
)
_bias_pipeline = pipeline(
    model="valurank/distilroberta-bias",
    device=get_device(), task=classification
)
_toxicity_pipeline = pipeline(
    model="unitary/toxic-bert",
    device=get_device(), task=classification
)
_spam_pipeline = pipeline(
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    device=get_device(),
    task=classification,
)
_advertisement_pipeline = pipeline(
    model="0x7o/roberta-base-ad-detector",
    device=get_device(), task=classification
)
_spacy_nlp = spacy.load("en_core_web_sm")


def is_unsafe(text: str) -> bool:
    results = _unsafe_pipeline(text)
    return any(res["label"] == "unsafe" and res["score"] > 0.5 for res in results)


def is_toxic(text: str) -> bool:
    results = _toxicity_pipeline(text)
    # Aggregate scores across toxicity labels; flag if any exceeds threshold
    return any(
        r["label"] in ["toxic", "severe_toxic", "obscene", "threat", "insult"]
        and r["score"] > 0.7
        for r in results
    )


def is_scam(text: str) -> bool:
    # Heuristic augmentation for scams (e.g., urgency, suspicious URLs)
    scam_patterns = [
        r"\b(buy|claim|win|free|offer|limited)\s+(now|today|urgent)",
        r"http[s]?://[^\s]+",  # Suspicious links
        r"\b(click|visit|call)\s+(here|below|now)",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in scam_patterns)


def is_advertisement(text: str) -> bool:
    result = _advertisement_pipeline(text)
    return any(res["label"] == "ad" and res["score"] > 0.5 for res in result)


def is_spammy(text: str) -> bool:
    result = _spam_pipeline(text)
    return any(res["label"] == "spam" and res["score"] > 0.5 for res in result)


def is_biased(text: str) -> bool:
    result = _bias_pipeline(text)
    return any(res["label"] == "biased" and res["score"] > 0.6 for res in result)


def has_sensitive_content(text: str) -> bool:
    doc = _spacy_nlp(text)
    sensitive_entities = ["PERSON", "EMAIL", "PHONE", "GPE", "SSN"]
    return any(ent.label_ in sensitive_entities for ent in doc.ents)


# To detect low-quality text

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

def is_low_quality(text: str) -> bool:
    text_list = replace_newlines(text)
    pred = _quality_model.predict(text_list, k=-1)
    for l, s in zip(*pred):
        score = 0
        for _l, _s in zip(l, s):
            score += score_dict[_l] * _s
        return float(score) < 1.0
    return False


# To detect known information

_known_info_model_name = "microsoft/Phi-4-mini-instruct"
_known_info_tokenizer = AutoTokenizer.from_pretrained(_known_info_model_name)
if _known_info_tokenizer.pad_token is None:
    _known_info_tokenizer.pad_token = _known_info_tokenizer.eos_token

_known_info_model = AutoModelForCausalLM.from_pretrained(
    _known_info_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def is_known_information(text: str) -> bool:
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
    return perplexity < threshold


# Code to use the above functions to classify text

def classify(text: str) -> list[str]:
    """Return a flag or "Good" if data has a high-quality."""
    tags = []
    if is_unsafe(text):
        tags.append("Unsafe")
    if is_toxic(text):
        tags.append("Toxic")
    if is_scam(text):
        tags.append("Scam")
    if is_advertisement(text):
        tags.append("Advertisement")
    if is_spammy(text):
        tags.append("Spammy")
    if is_biased(text):
        tags.append("Biased")
    if has_sensitive_content(text):
        tags.append("Sensitive Content")
    if is_low_quality(text):
        tags.append("Low Quality")
    if is_known_information(text):
        tags.append("Known Information")

    return tags


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
