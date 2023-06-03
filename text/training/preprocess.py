import pandas as pd
import os

from sklearn.model_selection import train_test_split

from ...manager.load_config import LOCAL

from ..webScraper.scrape import readUrlText

def loadAndFixDataset(path: str, save_path: str=os.path.join(LOCAL, "text", "data", "fixed_noticias_dataset.csv")):
    """
    Load and fix the dataset.

    Args:
        path (str): Path of the data file to load.
        save_path (str): Path to save the fixed dataset file.
            By default, it is saved in the 'fixed_noticias_dataset.csv' folder.

    Returns:
        pd.DataFrame: The fixed dataset.
    """
    df = pd.read_csv(path)
    text = []
    classification = []
    origin = []

    print(df.columns)

    for _, row in df.iterrows():
        # Add original data to the fixed dataset
        text.append(str(row["TEXTO"]).strip())
        classification.append(str(row["STRING_CLASSIFICATION"]).strip())
        origin.append(0)

        # Get text from a URL and add it to the fixed dataset
        aux = readUrlText(str(row["LINK"]).strip()).strip()
        
        if aux == "NOT FOUND":
            continue

        text.append(aux)
        classification.append(str(row["URL_CLASSIFICATION"]).strip())
        origin.append(1)

    # Create a new DataFrame with the fixed data
    new_df = pd.DataFrame({"TEXT": text, "CLASSIFICATION": classification, "origin": origin})

    # Save the new DataFrame as a CSV file
    new_df.to_csv(save_path)

    return new_df

def splitTrainTestEval(df: pd.DataFrame, save_dir: str=os.path.join(LOCAL, "text", "data", "tmp")):
    """
    Split the dataset into training, validation, and evaluation sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        save_dir (str): Directory to save the split datasets.
            By default, it is saved in the 'tmp' folder.

    Returns:
        tuple: A tuple containing the training, validation, and evaluation datasets.
    """
    # Split the dataset into training, validation, and evaluation sets
    train_val_df, eval_df = train_test_split(df, test_size=0.2)
    training_df, valid_df = train_test_split(train_val_df, test_size=0.2)

    print(f"EVALUATION DATASET: {len(eval_df)}")
    print(f"TRAINING DATASET: {len(training_df)}")
    print(f"VALIDATION DATASET: {len(valid_df)}")

    # Save the split datasets as CSV files
    training_df.to_csv(os.path.join(save_dir, "training_data.csv"))
    valid_df.to_csv(os.path.join(save_dir, "validation_data.csv"))
    eval_df.to_csv(os.path.join(save_dir, "evaluation_data.csv"))

    return training_df, valid_df, eval_df

def fixBuiltDataset(df):
    """
    Fix the dataset.

    Args:
        df (pd.DataFrame): The dataset to fix.

    Returns:
        pd.DataFrame: The fixed dataset.
    """
    # Remove rows with null values in the "TEXT" and "CLASSIFICATION" columns
    df = df[~df["TEXT"].isna()]
    df = df[~df["CLASSIFICATION"].isna()]
    
    # Replace incorrect values in the "CLASSIFICATION" column
    df["CLASSIFICATION"] = df["CLASSIFICATION"].replace(['NNGUNA', 'NINGUNO'], 'NINGUNA')
    df["CLASSIFICATION"] = df["CLASSIFICATION"].replace(['DEDFORESTACION'], 'DEFORESTACION')

    return df


if __name__ == "__main__":
    # Define the paths of the data files
    data_file = os.path.join(LOCAL, "text", "data", "noticias_dataset.csv")
    data_file_fixed = os.path.join(LOCAL, "text", "data", "fixed_noticias_dataset.csv")
    
    # Load and fix the dataset
    data_df = loadAndFixDataset(data_file)
    data_df = pd.read_csv(data_file_fixed)

    # Filter the data based on its origin (text or URL)
    txt_data_df = data_df[data_df["origin"] == 0]
    url_data_df = data_df[data_df["origin"] == 1]
    all_data_df = data_df
    
    # Fix the filtered datasets
    txt_data_df = fixBuiltDataset(txt_data_df)
    url_data_df = fixBuiltDataset(url_data_df)
    all_data_df = fixBuiltDataset(all_data_df)

    # Save the filtered datasets as CSV files
    txt_data_df.to_csv(os.path.join(LOCAL, "text", "data", "noticias_dataset_txt.csv"))
    url_data_df.to_csv(os.path.join(LOCAL, "text", "data", "noticias_dataset_url.csv"))
    all_data_df.to_csv(os.path.join(LOCAL, "text", "data", "noticias_dataset_all.csv"))
    
    # Split the datasets into training, validation, and evaluation sets
    train_df, valid_df, eval_df = splitTrainTestEval(txt_data_df, save_dir=os.path.join(LOCAL, "text", "data", "tmp_txt"))
    train_df, valid_df, eval_df = splitTrainTestEval(url_data_df, save_dir=os.path.join(LOCAL, "text", "data", "tmp_url"))
    train_df, valid_df, eval_df = splitTrainTestEval(all_data_df, save_dir=os.path.join(LOCAL, "text", "data", "tmp_all"))

