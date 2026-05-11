import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/deepglobe-land-cover-classification-dataset")
os.environ['DATASET_PATH'] = path

print("Path to dataset files:", path)