from flair.data import Sentence
from flair.models import SequenceTagger

# Cargar el modelo de etiquetador de secuencia preentrenado
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

def getEntities(s: str):
    """
    Obtener las entidades nombradas de una frase.

    Args:
        s (str): La frase de entrada.

    Returns:
        dict: Un diccionario que contiene las predicciones de las entidades.
            Las entidades se agrupan en las siguientes categorías: 'misc', 'dates', 'loc', 'org', 'per'.
    """
    # Crear un objeto de frase a partir del texto de entrada
    sent = Sentence(s)
    # Realizar la predicción de las entidades nombradas en la frase
    tagger.predict(sent)

    # Inicializar un diccionario para almacenar las predicciones de las entidades
    predictions_set = {"misc": set(), "dates": set(), "loc": set(), "org": set(), "per": set()}

    # Iterar sobre las entidades predichas en la frase
    for entity in sent.get_spans('ner'):
        if entity.text and entity.text != "":
            # Agregar el texto de la entidad al tipo correspondiente en el diccionario de predicciones

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
    # Ejemplo de uso: obtener las entidades nombradas de una frase
    print(getEntities("Hay algo que me gusta de ti Don Omar y ese algo me encanta en Colombia con personas de las Naciones Unidas y me vuelve loco volver a 1995. Luego en washington crearon arroz Chino."))
