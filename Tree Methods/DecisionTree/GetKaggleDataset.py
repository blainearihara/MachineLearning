# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 12:29:13 2025

@author: blain
"""
import os
import pandas as pd
import kagglehub

def get_dataset(dataset_id: str, filename: str) -> pd.DataFrame:
    """
    Ensures dataset is available (downloads if missing), then loads it.
    
    Args:
        dataset_id (str): Kaggle dataset ID (e.g. "arshid/iris-flower-dataset")
        filename (str): File to load within dataset (e.g. "Iris.csv")
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Download (cached after first time)
    path = kagglehub.dataset_download(dataset_id)
    
    # Build file path
    file_path = os.path.join(path, filename)
    
    # Check existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file {filename} not found in {path}")
    
    # Load into DataFrame
    return pd.read_csv(file_path)