# Amazon Deforestation Classifier

Authors: Joshua Smith, Evan Schmelkin

[Img](https://ecologi.com/resources/blog/industries-and-sectors-driving-deforestation-what-you-need-to-know)

A deep learning project that detects and tracks deforestation in the Amazon rainforest using satellite imagery.

## What It Does

The project has two parts:

1. **Trains a neural network** to identify forest vs. non-forest areas in satellite images
2. **Applies that model** to yearly satellite imagery of the Amazon (2015–2023) to measure how much forest has been lost over time

Results include a map showing predicted forest coverage for each year and a trend chart of deforestation percentage over time.

## How It Works

The model is a U-Net — a type of neural network commonly used for image segmentation — trained on the [DeepGlobe Land Cover dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset). It learns to look at a satellite image and label each pixel as either forest or not forest.

For the Amazon analysis, satellite imagery is pulled from [Google Earth Engine](https://earthengine.google.com/) using Sentinel-2 (a European Space Agency satellite with 10m resolution). The model then predicts forest coverage for each year, and deforestation is calculated as the percentage of pixels classified as non-forest.

## Project Structure

```
deforestation_detector.py        # Model definition, training, and prediction
amazon_analysis.py               # GEE pipeline: fetch images, run predictions, plot results
gee_captures/                    # Downloaded satellite images (auto-generated)
output/                          # Saved charts and maps (auto-generated)
deforestation_model_final.keras  # Saved trained model weights
```

## Setup

**Requirements:**
- Python 3.10+
- TensorFlow / Keras
- Google Earth Engine Python API (`earthengine-api`)
- A Google Earth Engine account and project ID

Install dependencies:
```bash
pip install tensorflow earthengine-api requests matplotlib
```

## Usage

**Step 1 — Train the model** (skip if `deforestation_model_final.keras` already exists):
```bash
python deforestation_detector.py
```
This trains the U-Net on the DeepGlobe dataset. Training takes a while depending on your hardware.

**Step 2 — Run the Amazon analysis:**

Edit the bottom of `amazon_analysis.py` to set your region and GEE project ID, then run:
```bash
python amazon_analysis.py
```

This will authenticate with Google Earth Engine, download Sentinel-2 imagery for each year, run the model, and save charts to the `output/` folder.

## Output

- `output/amazon_deforestation_maps.png` — Side-by-side satellite image and predicted forest mask for each year
- `output/amazon_deforestation_trend.png` — Line chart of deforestation percentage over time

## Notes

- The model was trained on high-resolution commercial satellite imagery, so predictions on Sentinel-2 imagery use an adjusted confidence threshold to account for differences between the two image types. Fine-tuning on labeled Sentinel-2 data (e.g., the [Amazon Forest Zenodo dataset](https://zenodo.org/records/4498086)) would improve accuracy.
- Sentinel-2 data availability starts in 2015, so early years may be skipped depending on the region.
- The analyzed region can be changed by editing `AMAZON_BBOX` in `amazon_analysis.py`. Coordinates are `[west, south, east, north]` in decimal degrees.

## Citation

```
@InProceedings{DeepGlobe18,
  author    = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and
               Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
  title     = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2018}
}
```
