import pandas as pd
import os

from sklearn.model_selection import train_test_split

from ...manager.load_config import LOCAL

from ..webScraper.scrape import readUrlText

def loadAndFixDataset(path: str, save_path: str=os.path.join(LOCAL, "text", "data", "fixed_noticias_dataset.csv")):
    """
    Carga y arregla el conjunto de datos.
    
    Args:
        path (str): Ruta del archivo de datos a cargar.
        save_path (str): Ruta para guardar el archivo de datos arreglado. 
            Por defecto, se guarda en la carpeta 'fixed_noticias_dataset.csv'.

    Returns:
        pd.DataFrame: El conjunto de datos arreglado.
    """
    df = pd.read_csv(path)
    text = []
    classification = []
    origin = []

    print(df.columns)

    for _, row in df.iterrows():
        # Agregar los datos originales al conjunto de datos arreglado
        text.append(str(row["TEXTO"]).strip())
        classification.append(str(row["STRING_CLASSIFICATION"]).strip())
        origin.append(0)

        # Obtener el texto de una URL y agregarlo al conjunto de datos arreglado
        aux = readUrlText(str(row["LINK"]).strip()).strip()
        
        if aux == "NOT FOUND":
            continue

        text.append(aux)
        classification.append(str(row["URL_CLASSIFICATION"]).strip())
        origin.append(1)

    # Crear un nuevo DataFrame con los datos arreglados
    new_df = pd.DataFrame({"TEXT": text, "CLASSIFICATION": classification, "origin": origin})

    # Guardar el nuevo DataFrame como archivo CSV
    new_df.to_csv(save_path)

    return new_df

def splitTrainTestEval(df: pd.DataFrame, save_dir: str=os.path.join(LOCAL, "text", "data", "tmp")):
    """
    Divide el conjunto de datos en entrenamiento, validación y evaluación.

    Args:
        df (pd.DataFrame): El conjunto de datos a dividir.
        save_dir (str): Directorio para guardar los conjuntos de datos divididos.
            Por defecto, se guarda en la carpeta 'tmp'.

    Returns:
        tuple: Una tupla con los conjuntos de datos de entrenamiento, validación y evaluación.
    """
    # Dividir el conjunto de datos en entrenamiento, validación y evaluación
    train_val_df, eval_df = train_test_split(df, test_size=0.2)
    training_df, valid_df = train_test_split(train_val_df, test_size=0.2)

    print(f"EVALUATION DATASET: {len(eval_df)}")
    print(f"TRAINING DATASET: {len(training_df)}")
    print(f"VALIDATION DATASET: {len(valid_df)}")

    # Guardar los conjuntos de datos divididos como archivos CSV
    training_df.to_csv(os.path.join(save_dir, "training_data.csv"))
    valid_df.to_csv(os.path.join(save_dir, "validation_data.csv"))
    eval_df.to_csv(os.path.join(save_dir, "evaluation_data.csv"))

    return training_df, valid_df, eval_df

def fixBuiltDataset(df):
    """
    Arregla el conjunto de datos.

    Args:
        df (pd.DataFrame): El conjunto de datos a arreglar.

    Returns:
        pd.DataFrame: El conjunto de datos arreglado.
    """
    # Eliminar filas con valores nulos en las columnas "TEXT" y "CLASSIFICATION"
    df = df[~df["TEXT"].isna()]
    df = df[~df["CLASSIFICATION"].isna()]
    
    # Reemplazar valores incorrectos en la columna "CLASSIFICATION"
    df["CLASSIFICATION"] = df["CLASSIFICATION"].replace(['NNGUNA', 'NINGUNO'], 'NINGUNA')
    df["CLASSIFICATION"] = df["CLASSIFICATION"].replace(['DEDFORESTACION'], 'DEFORESTACION')

    return df


if __name__ == "__main__":
    # Definir las rutas de los archivos de datos
    data_file = os.path.join(LOCAL, "text", "data", "noticias_dataset.csv")
    data_file_fixed = os.path.join(LOCAL, "text", "data", "fixed_noticias_dataset.csv")
    
    # Cargar y arreglar el conjunto de datos
    data_df = loadAndFixDataset(data_file)
    data_df = pd.read_csv(data_file_fixed)

    # Filtrar los datos según su origen (texto o URL)
    txt_data_df = data_df[data_df["origin"] == 0]
    url_data_df = data_df[data_df["origin"] == 1]
    all_data_df = data_df
    
    # Arreglar los conjuntos de datos filtrados
    txt_data_df = fixBuiltDataset(txt_data_df)
    url_data_df = fixBuiltDataset(url_data_df)
    all_data_df = fixBuiltDataset(all_data_df)

    # Guardar los conjuntos de datos filtrados como archivos CSV
    txt_data_df.to_csv(os.path.join(LOCAL, "text", "data", "noticias_dataset_txt.csv"))
    url_data_df.to_csv(os.path.join(LOCAL, "text", "data", "noticias_dataset_url.csv"))
    all_data_df.to_csv(os.path.join(LOCAL, "text", "data", "noticias_dataset_all.csv"))
    
    # Dividir los conjuntos de datos en entrenamiento, validación y evaluación
    train_df, valid_df, eval_df = splitTrainTestEval(txt_data_df, save_dir=os.path.join(LOCAL, "text", "data", "tmp_txt"))
    train_df, valid_df, eval_df = splitTrainTestEval(url_data_df, save_dir=os.path.join(LOCAL, "text", "data", "tmp_url"))
    train_df, valid_df, eval_df = splitTrainTestEval(all_data_df, save_dir=os.path.join(LOCAL, "text", "data", "tmp_all"))

