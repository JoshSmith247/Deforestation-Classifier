import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/deepglobe-land-cover-classification-dataset")

print("Path to dataset files:", path)