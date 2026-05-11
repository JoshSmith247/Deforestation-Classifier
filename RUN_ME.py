import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import install
from deforestation_detector import DeforestationClassifier
from amazon_analysis import analyze

DATA_DIR = os.environ['DATASET_PATH'] + '/train'
classifier = DeforestationClassifier(data_dir=DATA_DIR)
classifier.get()

analyze(
    bbox=[-55.5, -7.6, -55.4, -7.5], # [-62.55, -10.85, -62.45, -10.75] # High deforestation demo
    years=list(range(2015, 2024)),
    gee_project='amazon-analysis-492817',
    output_dir='blog_figures',
)
