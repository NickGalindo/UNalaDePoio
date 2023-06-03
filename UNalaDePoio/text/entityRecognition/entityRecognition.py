from flair.data import Sentence
from flair.models import SequenceTagger

# Load the pre-trained sequence tagger model
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

def getEntities(s: str):
    """
    Get named entities from a sentence.

    Args:
        s (str): The input sentence.

    Returns:
        dict: A dictionary containing the predicted entities.
            Entities are grouped into the following categories: 'misc', 'dates', 'loc', 'org', 'per'.
    """
    # Create a sentence object from the input text
    sent = Sentence(s)
    # Predict the named entities in the sentence
    tagger.predict(sent)

    # Initialize a dictionary to store the entity predictions
    predictions_set = {"misc": set(), "dates": set(), "loc": set(), "org": set(), "per": set()}

    # Iterate over the predicted entities in the sentence
    for entity in sent.get_spans('ner'):
        if entity.text and entity.text != "":
            # Add the entity text to the corresponding type in the predictions dictionary
            if entity.tag == "PERSON":
                predictions_set["per"].add(entity.text)
            elif entity.tag == "GPE":
                predictions_set["loc"].add(entity.text)
            elif entity.tag == "LOC":
                predictions_set["loc"].add(entity.text)
            elif entity.tag == "ORG":
                predictions_set["org"].add(entity.text)
            elif entity.tag == "DATE":
                predictions_set["dates"].add(entity.text)
            else:
                predictions_set["misc"].add(entity.text)

    predictions = {key: list(predictions_set[key]) for key in predictions_set.keys()}
    return predictions




if __name__ == "__main__":
    # Example usage: get named entities from a sentence
    print(getEntities("Hay algo que me gusta de ti Don Omar y ese algo me encanta en Colombia con personas de las Naciones Unidas y me vuelve loco volver a 1995. Luego en washington crearon arroz Chino."))
