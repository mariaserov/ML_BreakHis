import kagglehub
import os
import numpy as np
import pandas as pd

# Get data from kaggle - use these two lines to get BreakHis data!

path = kagglehub.dataset_download("ambarish/breakhis")
print("Path to dataset files:", path)
print("Run this to copy them into the data folder (alternatively, create a symlink):\n", "cp -r", path[:-2], "../data/")

# Load data - you don't need to do it if you're just getting BreakHis data

data_dir = "../data/versions/4/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
metadata = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".png"):
            # Extract label from the folder structure
            label = "malignant" if "malignant" in root else "benign"
            
            # Extract magnification 
            
            magnification = None
            for part in root.split(os.sep):
                if part.endswith("X") and part[:-1].isdigit(): 
                    magnification = part
                    break
            
            # Extract tumor subtype 
            tumor_subtype = None
            for part in root.split(os.sep):
                if part in ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",  # Benign subtypes
                           "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]:  # Malignant subtypes
                    tumor_subtype = part
                    break
            
            # Append filepath, label, magnification, and tumor subtype to metadata
            metadata.append((os.path.join(root, file), label, magnification, tumor_subtype))

# Convert to DataFrame
df = pd.DataFrame(metadata, columns=["filepath", "label", "magnification", "tumor_subtype"])

df.to_csv("../data/metadata.csv")

# Debugging: Check the shape and first few rows of the DataFrame
print(f"DataFrame shape: {df.shape}")
print(df.head())