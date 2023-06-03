from .text import textPredict, urlPredict
import json

def ner_from_str(text: str, output_path: str):
    res = textPredict(text)

    res_json = json.dumps(res)

    with open(output_path, "w") as outfile:
        outfile.write(res_json)

    return res_json

def ner_from_file(text_path: str, output_path: str):
    with open(text_path, "r") as inpfile:
        text = inpfile.read()

    res = textPredict(text)

    res_json = json.dumps(res)

    with open(output_path, "w") as outfile:
        outfile.write(res_json)

    return res_json

def ner_from_url(url: str, output_path: str):
    res = urlPredict(url)

    res_json = json.dumps(res)

    with open(output_path, "w") as outfile:
        outfile.write(res_json)

    return res_json
