from typing import Any, Dict
from .training.loadModel import loadModel, loadTokenizers
from .entityRecognition import entityRecognition
from .training.trainBert import label_encoding
from .webScraper.scrape import readUrlText
import torch

txt_model, url_model = loadModel()
txt_tokenizer, url_tokenizer = loadTokenizers()

reverse_label_encodings = {label_encoding[i]: i for i in label_encoding.keys()}

def textPredict(text):
    full_preds: Dict[str, Any] = entityRecognition.getEntities(text)

    encoding = txt_tokenizer([text], truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        logits = txt_model(**encoding).logits #type: ignore

    class_pred = reverse_label_encodings[logits.argmax().item()]
    
    full_preds["impact"] = class_pred
    full_preds["text"] = text

    return full_preds


def urlPredict(url):
    text = readUrlText(url.strip())

    if text == "NOT FOUND":
        return {"text": [], "impact": "NINGUNA", "misc": [], "dates": [], "loc": [], "org": [], "per": []}

    full_preds: Dict[str, Any] = entityRecognition.getEntities(text)

    encoding = url_tokenizer([text], truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        logits = url_model(**encoding).logits #type: ignore

    class_pred = reverse_label_encodings[logits.argmax().item()]
    
    full_preds["impact"] = class_pred
    full_preds["text"] = text

    return full_preds
