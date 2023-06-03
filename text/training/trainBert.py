from typing import Any, Dict
# base data science imports
import numpy as np
import pandas as pd

# transformers imports
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.models.bert.modeling_bert import BertForSequenceClassification

import torch
import torch.utils.data

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os

from ...manager.load_config import LOCAL

# Definir un diccionario para el mapeo de etiquetas
label_encoding = {'NINGUNA':0, 'MINERIA':1, 'DEFORESTACION':2, 'CONTAMINACION':3}

# Definir una clase para el conjunto de datos
class DocDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        """
        Inicializa un conjunto de datos para el modelo.

        :param encodings: Los encodings de entrada.
        :param labels: Las etiquetas de los datos.
        """
        self.encodings = encodings
        self.labels = [label_encoding[i] for i in labels]

    def __getitem__(self, idx):
        """
        Obtiene un elemento del conjunto de datos.

        :param idx: El índice del elemento.
        :return: El elemento en la posición `idx`.
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Obtiene la longitud del conjunto de datos.

        :return: La longitud del conjunto de datos.
        """
        return len(self.labels)

def __compute_metrics(pred):
    """
    Calcula las métricas de evaluación para un conjunto de predicciones.

    :param pred: El objeto de predicciones.
    :return: Un diccionario con las métricas calculadas.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def __preprocessDataframeDatasets(
        training_df: pd.DataFrame, 
        eval_df: pd.DataFrame, 
        valid_df: pd.DataFrame):
    """
    Preprocesa los datos del archivo y los convierte en conjuntos de datos.

    :param training_df: el dataframe de entrenamiento con los datos correspondientes
    :param eval_df: el dataframe de evaluación con los datos correspondientes
    :param valid_df: el dataframe de validación con los datos correspondientes
    :return: un diccionario con los conjuntos de datos preprocesados
    """


    # Convertir los objetos del dataframe en listas
    train_text = training_df["TEXT"].tolist()
    train_label = training_df["CLASSIFICATION"].tolist()
    validation_text = valid_df["TEXT"].tolist()
    validation_label = valid_df["CLASSIFICATION"].tolist()
    test_text = eval_df["TEXT"].tolist()
    test_label = eval_df["CLASSIFICATION"].tolist()

    # Devolver un mapa con las listas de datos
    return {
        "train_text": train_text,
        "train_label": train_label,
        "validation_text": validation_text,
        "validation_label": validation_label,
        "test_text": test_text,
        "test_label": test_label
    }

def bertBaseSpanishUncasedFinetune(
        training_df: pd.DataFrame,
        evaluation_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        train_epochs: int=40, 
        batch_size: int=64, 
        learning_rate: float=5e-5,
        save_dir: str=os.path.join(LOCAL, "text", "data", "tmp")):

    """
    Ejecuta bert base spanish uncased

    :param training_df: el dataframe de entrenamiento con los datos correspondientes
    :param evaluation_df: el dataframe de evaluación con los datos correspondientes
    :param validation_df: el dataframe de validación con los datos correspondientes
    :param train_epochs: la cantidad de épocas para entrenar
    :param batch_size: el tamaño del lote para ejecutar el entrenamiento
    :param learning_rate: la tasa de aprendizaje para ejecutar el entrenamiento
    """

    # Cargar los datos del archivo y preprocesarlos
    tmp_dataset_dict = __preprocessDataframeDatasets(
            training_df=training_df,
            eval_df=evaluation_df,
            valid_df=validation_df,
    )
    

    # Ejecutar el entrenamiento de bert
    return __runBertTraining(
        model_name="dccuchile/bert-base-spanish-wwm-uncased",
        data_stream=tmp_dataset_dict,
        train_epochs=train_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir
    )


def __runBertTraining(
        model_name: Any, 
        data_stream: Dict, 
        train_epochs: int, 
        batch_size: int, 
        learning_rate: float,
        save_dir: str):
    """
    Ejecuta un entrenamiento de BERT con un modelo, un tokenizador y un flujo de datos específicos.

    :param model_name: el nombre del modelo en el que se entrenará
    :param data_stream: los datos a utilizar
    :param train_epochs: las épocas de entrenamiento a ejecutar
    :param batch_size: el tamaño del lote para ejecutar
    :param learning_rate: la tasa de aprendizaje a utilizar
    :param save_dir: el directorio para guardar el modelo
    :return: el modelo y el tokenizador entrenados
    """

    # Cargar el tokenizer y el modelo para el entrenamiento desde un modelo pre-entrenado
    print("LOADING TOKENIZER AND MODEL")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
    print("FINISHED LOADING TOKENIZER AND MODEL")

    # Tokenizar las listas de datos de entrenamiento, validación y prueba
    print("STARTING TOKENIZATION")
    train_encodings = tokenizer(data_stream["train_text"], truncation=True, padding=True, return_tensors="pt")
    validation_encodings = tokenizer(data_stream["validation_text"], truncation=True, padding=True, return_tensors="pt")
    test_encodings = tokenizer(data_stream["test_text"], truncation=True, padding=True, return_tensors="pt")
    print("TRAINING, VALIDATION, AND TEST ENCODINGS TOKENIZED")

    # Convertir las codificaciones en un conjunto de datos agregando las etiquetas para ser utilizadas de manera iterativa
    train_dataset = DocDataset(train_encodings, data_stream["train_label"])
    val_dataset = DocDataset(validation_encodings, data_stream["validation_label"])
    test_dataset = DocDataset(test_encodings, data_stream["test_label"])
    
    # Establecer el directorio de salida del modelo
    output_dir = os.path.join(save_dir, "model", model_name.split("/")[-1], f"results(ep:{train_epochs}, bs:{batch_size}, lr:{learning_rate})")
    logging_dir = os.path.join(save_dir, "logging", model_name.split("/")[-1], f"results(ep:{train_epochs}, bs:{batch_size}, lr:{learning_rate})")
    tokenizer_dir = os.path.join(save_dir, "tokenizer", model_name.split("/")[-1], f"results(ep:{train_epochs}, bs:{batch_size}, lr:{learning_rate})")

    # Ejecutar el modelo en cuda y borrar la caché por si acaso
    model.cuda() #type: ignore
    torch.cuda.empty_cache()
    print("INITIATED CUDA")

    print(batch_size)

    # Make a training Arguments class with all of the arguments to be used for the model training
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=train_epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        learning_rate=learning_rate,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logging_dir,            # directory for storing logs
        logging_steps=10,
        logging_strategy="no",
        do_eval=True,
        fp16=True
    )
    print("FINISHED SETTING THE TRAINING ARGUMENTS")

    # Initialize the Trainer and start training
    trainer = Trainer(
        model=model, #type: ignore
        args=training_args,
        compute_metrics=__compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    print("INITIALIZED TRAINING")

    trainer.train()
    print("FINISHED TRAINING")

    # Start the evaluation of the trained model 
    evaluation = trainer.evaluate()
    print("\n"*5)
    print(f"PRINTING TRAINER RESULTS: {evaluation}")

    # test the prediction capabilities of the model
    prediction = trainer.predict(test_dataset=test_dataset)
    print("\n"*3)
    print(prediction)

    # save the tokenizer
    tokenizer.save_pretrained(tokenizer_dir)
    trainer.save_model(os.path.join(output_dir, "finalModel"))

    return trainer.model, tokenizer

if __name__ == "__main__":
    train_path = os.path.join(LOCAL, "text", "data", "tmp_all", "training_data.csv")
    valid_path = os.path.join(LOCAL, "text", "data", "tmp_all", "validation_data.csv")
    eval_path = os.path.join(LOCAL, "text", "data", "tmp_all", "evaluation_data.csv")
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    valid_df = pd.read_csv(valid_path)
    
    bertBaseSpanishUncasedFinetune(train_df, eval_df, valid_df, train_epochs=40, batch_size=16, save_dir=os.path.join(LOCAL, "text", "data", "all_model"))
