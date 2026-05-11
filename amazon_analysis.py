import ee
import requests
import os
import matplotlib.pyplot as plt

from deforestation_detector import DeforestationClassifier

# --- 1. AUTHENTICATE & INITIALIZE ---

def init_gee(project: str):
    """Authenticate and initialize Google Earth Engine."""
    ee.Authenticate()
    ee.Initialize(project=project)

# --- 2. FETCH IMAGES ---

def _mask_s2_clouds(image):
    """Mask clouds and cirrus using Sentinel-2's QA60 band."""
    qa = image.select('QA60')
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)   # opaque clouds
          .And(qa.bitwiseAnd(1 << 11).eq(0))  # cirrus
    )
    return image.updateMask(mask)


def fetch_images(bbox: list[float], years: list[int], output_dir: str = 'gee_captures') -> dict[int, str]:
    """
    Download one cloud-free Sentinel-2 composite image per year for the given bounding box.

    Args:
        bbox:       [west, south, east, north] in lon/lat degrees
        years:      list of years to fetch (e.g. [2015, 2016, ..., 2023])
        output_dir: folder to save downloaded images

    Returns:
        dict mapping year -> local file path
    """
    os.makedirs(output_dir, exist_ok=True)
    region = ee.Geometry.Rectangle(bbox)
    paths = {}

    for year in years:
        filepath = os.path.join(output_dir, f'amazon_{year}.jpg')

        if os.path.exists(filepath):
            print(f"[{year}] Already downloaded, skipping.")
            paths[year] = filepath
            continue

        # Dry season composite (June–October); per-pixel QA masking handles residual clouds
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(f'{year}-06-01', f'{year}-10-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
            .map(_mask_s2_clouds)
            .sort('CLOUDY_PIXEL_PERCENTAGE')
            .limit(100)
        )

        count = collection.size().getInfo()
        if count == 0:
            print(f"[{year}] No images found — skipping.")
            continue

        # Median composite across the masked scenes; unmask fills any remaining gaps with 0
        image = collection.median().unmask(0)

        # Sentinel-2 SR bands: B4=Red (10m), B3=Green (10m), B2=Blue (10m)
        # SR values are scaled 0–10000 (reflectance × 10000)
        rgb = image.select(['B4', 'B3', 'B2'])

        url = rgb.getThumbURL({
            'region': region,
            'dimensions': 512,   # download at 512 then model resizes to 256
            'format': 'jpg',
            'min': 0,
            'max': 3000,   # ~30% reflectance covers forest to bright bare soil
            'gamma': 1.4,
        })

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)

        paths[year] = filepath
        print(f"[{year}] Downloaded -> {filepath}")

    return paths

# --- 3. ANALYZE ---

def analyze(
    bbox: list[float],
    years: list[int],
    model_path: str = 'deforestation_model_final.keras',
    gee_project: str = 'your-gee-project-id',
):
    """
    Full pipeline: authenticate GEE, fetch images, run predictions, plot results.

    Args:
        bbox:        [west, south, east, north] in lon/lat degrees
        years:       list of years to analyze
        model_path:  path to saved .keras model
        gee_project: your Google Earth Engine project ID
    """
    init_gee(gee_project)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run deforestation-detector.py first to train and save the model."
        )

    # Load trained model (data_dir unused — only needed during training)
    classifier = DeforestationClassifier(data_dir='', save_path=model_path)
    classifier.get()

    image_paths = fetch_images(bbox, years)
    if not image_paths:
        print("No images to analyze.")
        return

    # Run predictions
    results = {}
    for year in sorted(image_paths.keys()):
        mask = classifier.predict(image_paths[year], normalize=True)
        deforestation_pct = (1 - mask.mean()) * 100
        results[year] = {
            'path': image_paths[year],
            'mask': mask,
            'deforestation_pct': deforestation_pct,
        }
        print(f"[{year}] Deforestation: {deforestation_pct:.1f}%")

    os.makedirs('blog_figures', exist_ok=True)
    _plot_maps(results)
    _plot_trend(results)

    return results

# --- 4. VISUALIZE ---

def _plot_maps(results: dict):
    """Side-by-side satellite image and predicted forest mask for each year."""
    years = sorted(results.keys())
    n = len(years)

    _, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for i, year in enumerate(years):
        r = results[year]
        img = plt.imread(r['path'])

        axes[i][0].imshow(img)
        axes[i][0].set_title(f'{year}  —  Satellite (RGB)')
        axes[i][0].axis('off')

        axes[i][1].imshow(r['mask'], cmap='Greens_r')
        axes[i][1].set_title(f'{year}  —  Forest Mask\nDeforestation: {r["deforestation_pct"]:.1f}%')
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.savefig('blog_figures/amazon_deforestation_maps.png', dpi=150, bbox_inches='tight')
    print("Saved: amazon_deforestation_maps.png")
    plt.show()


def _plot_trend(results: dict):
    """Line chart of deforestation % over time."""
    years = sorted(results.keys())
    pcts  = [results[y]['deforestation_pct'] for y in years]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, pcts, 'o-', color='sienna', linewidth=2.5, markersize=8, label='Deforestation %')
    ax.fill_between(years, pcts, alpha=0.15, color='sienna')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Deforestation (%)', fontsize=12)
    ax.set_title('Amazon Deforestation Over Time', fontsize=14)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('blog_figures/amazon_deforestation_trend.png', dpi=150, bbox_inches='tight')
    print("Saved: amazon_deforestation_trend.png")
    plt.show()

# --- RUN ---

if __name__ == '__main__':
    # Rondônia, Brazil — one of the most heavily deforested areas of the Amazon
    # AMAZON_BBOX = [-65.0, -13.0, -59.0, -8.0]  # [west, south, east, north]
    # AMAZON_BBOX = [-63.0, -11.5, -62.0, -10.5] # Smaller region
    # AMAZON_BBOX = [-62.45, -10.95, -62.35, -10.85] # Best so far
    # AMAZON_BBOX = [-62.425, -10.425, -62.375, -10.375]
    # AMAZON_BBOX = [-62.41, -10.41, -62.39, -10.39]
    # Other
    # AMAZON_BBOX = [-60.1, -3.2, -60.0, -3.1]
    AMAZON_BBOX = [-55.5, -7.6, -55.4, -7.5] # Good second option
    # AMAZON_BBOX = [-52.5, -13.1, -52.4, -13.0]
    # AMAZON_BBOX = [-62.55, -10.85, -62.45, -10.75] # General High Deforestation
    YEARS       = list(range(2015, 2024))
    GEE_PROJECT = 'amazon-analysis-492817'

    analyze(AMAZON_BBOX, YEARS, gee_project=GEE_PROJECT)
