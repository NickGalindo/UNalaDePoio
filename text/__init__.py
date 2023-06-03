from .text import textPredict, urlPredict
import json

def ner_from_str(text: str, output_path: str):
    """
    Predicts named entities and impact from a given string of text and saves the result as JSON.
    
    Args:
        text (str): The input text.
        output_path (str): The path to save the JSON output.
    
    Returns:
        str: The result as a JSON string.
    """
    # Predict named entities and impact from the text
    res = textPredict(text)

    # Convert the result to a JSON string
    res_json = json.dumps(res)

    # Write the JSON string to the output file
    with open(output_path, "w") as outfile:
        outfile.write(res_json)

    return res_json

def ner_from_file(text_path: str, output_path: str):
    """
    Predicts named entities and impact from a file containing text and saves the result as JSON.
    
    Args:
        text_path (str): The path to the input text file.
        output_path (str): The path to save the JSON output.
    
    Returns:
        str: The result as a JSON string.
    """
    # Read the text from the input file
    with open(text_path, "r") as inpfile:
        text = inpfile.read()
        
    # Predict named entities and impact from the text
    res = textPredict(text)
    
    # Convert the result to a JSON string
    res_json = json.dumps(res)

    # Write the JSON string to the output file
    with open(output_path, "w") as outfile:
        outfile.write(res_json)

    return res_json

def ner_from_url(url: str, output_path: str):
    """
    Predicts named entities and impact from a given URL and saves the result as JSON.
    
    Args:
        url (str): The input URL.
        output_path (str): The path to save the JSON output.
    
    Returns:
        str: The result as a JSON string.
    """

    # Predict named entities and impact from the URL
    res = urlPredict(url)
    
    # Convert the result to a JSON string
    res_json = json.dumps(res)

    # Write the JSON string to the output file
    with open(output_path, "w") as outfile:
        outfile.write(res_json)

    return res_json
