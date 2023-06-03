from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from ...manager.load_config import LOCAL

import os

def loadTokenizers(txt_tok_path: str=os.path.join(LOCAL, "text", "data", "txt_model", "tokenizer", "bert-base-spanish-wwm-uncased", "results(ep:40, bs:32, lr:5e-05)"), url_tok_path: str=os.path.join(LOCAL, "text", "data", "url_model", "tokenizer", "bert-base-spanish-wwm-uncased", "results(ep:40, bs:16, lr:5e-05)") ):
    txt_tokenizer = BertTokenizer.from_pretrained(txt_tok_path)
    url_tokenizer = BertTokenizer.from_pretrained(url_tok_path)

    return txt_tokenizer, url_tokenizer

def loadModel(txt_model_path: str=os.path.join(LOCAL, "text", "data", "url_model", "model", "bert-base-spanish-wwm-uncased", "results(ep:40, bs:16, lr:5e-05)", "finalModel"), url_model_path: str=os.path.join(LOCAL, "text", "data", "url_model", "model", "bert-base-spanish-wwm-uncased", "results(ep:40, bs:16, lr:5e-05)", "finalModel") ):
    txt_model = BertForSequenceClassification.from_pretrained(txt_model_path)
    url_model = BertForSequenceClassification.from_pretrained(url_model_path)

    return txt_model, url_model
