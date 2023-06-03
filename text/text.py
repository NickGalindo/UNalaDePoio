from typing import Any, Dict
from .training.loadModel import loadModel, loadTokenizers
from .entityRecognition import entityRecognition
from .training.trainBert import label_encoding
from .webScraper.scrape import readUrlText
import torch

# Load the text and URL models
txt_model, url_model = loadModel()

# Load the text and URL tokenizers
txt_tokenizer, url_tokenizer = loadTokenizers()

# Create a reverse mapping of label encodings for decoding predictions
reverse_label_encodings = {label_encoding[i]: i for i in label_encoding.keys()}

def textPredict(text):
    """
    Predicts entities and impact from given text.
    
    Args:
        text (str): The input text.
    
    Returns:
        Dict[str, Any]: A dictionary containing the predicted entities and impact.
    """
    full_preds: Dict[str, Any] = entityRecognition.getEntities(text)

    # Encode the text using the text tokenizer
    encoding = txt_tokenizer([text], truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        # Get the logits from the text model
        logits = txt_model(**encoding).logits #type: ignore

    # Get the predicted class based on the logits
    class_pred = reverse_label_encodings[logits.argmax().item()]

    # Add the predicted impact and text to the dictionary
    full_preds["impact"] = class_pred
    full_preds["text"] = text

    return full_preds


def urlPredict(url):
    """
    Predicts entities and impact from the text extracted from a given URL.
    
    Args:
        url (str): The input URL.
    
    Returns:
        Dict[str, Any]: A dictionary containing the predicted entities and impact.
    """

    # Extract the text from the URL
    text = readUrlText(url.strip()) 
    
    # If text is not found, return default predictions
    if text == "NOT FOUND":
        return {"text": [], "impact": "NINGUNA", "misc": [], "dates": [], "loc": [], "org": [], "per": []}

    full_preds: Dict[str, Any] = entityRecognition.getEntities(text)

    # Encode the text using the URL tokenizer
    encoding = url_tokenizer([text], truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        # Get the logits from the URL model
        logits = url_model(**encoding).logits #type: ignore

    # Get the predicted class based on the logits
    class_pred = reverse_label_encodings[logits.argmax().item()]

    # Add the predicted impact and text to the dictionary
    full_preds["impact"] = class_pred
    full_preds["text"] = text

    return full_preds
