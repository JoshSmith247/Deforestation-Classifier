import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
DATA_DIR = '/Users/jsmith/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2/train'

print("Loading Data...")

def load_deepglobe_data(base_path):
    # Load training satellite images
    sat_images = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('_sat.jpg')])
    # Load training mask images
    mask_images = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('_mask.png')])
    
    return sat_images, mask_images

# Update this path to your local directory
data_path = '/Users/jsmith/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2/train'
image_paths, mask_paths = load_deepglobe_data(data_path)

print(f"Found {len(image_paths)} image-mask pairs.")

# Create a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

def process_path(image_path, mask_path):
    # Load Image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256]) / 255.0
    
    # Load Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, [256, 256], method='nearest')
    
    # CONVERT TO BINARY: DeepGlobe Forest is [0, 255, 0]
    # We look for pixels where Green > Red and Green > Blue
    forest_mask = tf.logical_and(mask[:,:,1] > mask[:,:,0], mask[:,:,1] > mask[:,:,2])
    forest_mask = tf.cast(forest_mask, tf.float32)
    forest_mask = tf.expand_dims(forest_mask, -1) # Shape (256, 256, 1)
    
    return img, forest_mask

# Map the processing function over the dataset
dataset = dataset.map(process_path).batch(16)

print("Defining the model...")

def simple_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Downsample (Encoder)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D()(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D()(conv2)

    # Bridge
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)

    # Upsample (Decoder)
    up1 = layers.UpSampling2D()(conv3)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    
    up2 = layers.UpSampling2D()(conv4)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(up2)

    # Output: 256x256x1 with sigmoid (confidence per pixel)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    return models.Model(inputs, outputs)

model = simple_unet()

print("Compiling...")

# Split the data first so it isn't validating on training data
num_batches = len(image_paths) // 16
train_size = int(0.8 * num_batches)
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Starting training...")

history = model.fit(train_ds, validation_data=val_ds, epochs=1)

print("Running on test images...")

images, masks = next(iter(val_ds))
preds = model.predict(images)

# Visualize multiple test images
NUM_IMAGES = min(4, len(images))
fig, axes = plt.subplots(NUM_IMAGES, 3, figsize=(12, 4 * NUM_IMAGES))

for i in range(NUM_IMAGES):
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title(f"Satellite Image {i+1}")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(masks[i], cmap='Greens')
    axes[i, 1].set_title(f"Actual Forest Mask {i+1}")
    axes[i, 1].axis('off')

    im = axes[i, 2].imshow(preds[i], cmap='RdYlGn')  # Red = Low Confidence, Green = High
    axes[i, 2].set_title(f"Model Confidence Heatmap {i+1}")
    axes[i, 2].axis('off')
    fig.colorbar(im, ax=axes[i, 2])

plt.tight_layout()
plt.show()