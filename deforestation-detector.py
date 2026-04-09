import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

# --- CONFIGURATION ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'deforestation_model_final.keras'
DATA_DIR = '/kaggle/input/deepglobe-land-cover-classification-dataset/train'

if os.path.exists(MODEL_SAVE_PATH):
    os.remove(MODEL_SAVE_PATH)
    print("Old model removed. Starting fresh.")

# --- DeepGlobe label palette ---
# (0,255,0)   = Forest        ← what we want
# (255,255,0) = Agriculture
# (0,255,255) = Water
# (255,255,255)= Urban/barren
# (255,0,255) = Rangeland
# (255,0,0)   = Unknown/cloud

# --- PATH LOADER ---
def load_paths(base_path):
    sat_images  = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('_sat.jpg')])
    mask_images = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('_mask.png')])
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(sat_images))
    return [sat_images[i] for i in indices], [mask_images[i] for i in indices]

# --- COMPUTE CLASS WEIGHT from a sample of masks ---
def compute_pos_weight(mask_paths, n=100):
    """Returns pos_weight = (non_forest_pixels / forest_pixels) for weighted BCE."""
    forest_total = 0
    nonforest_total = 0
    for mp in mask_paths[:n]:
        mask = tf.io.read_file(mp)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
        m = tf.cast(mask, tf.float32)
        forest = tf.logical_and(
            tf.logical_and(m[:,:,1] > 200, m[:,:,0] < 50), m[:,:,2] < 50
        )
        f = tf.reduce_sum(tf.cast(forest, tf.float32)).numpy()
        forest_total    += f
        nonforest_total += (IMG_SIZE[0] * IMG_SIZE[1]) - f
    pos_weight = nonforest_total / max(forest_total, 1.0)
    print(f"Computed pos_weight from {n} masks: {pos_weight:.2f}  "
          f"(forest={forest_total/(forest_total+nonforest_total)*100:.1f}% of pixels)")
    return pos_weight

# --- DATA PIPELINE ---
def process_path(image_path, mask_path, augment=False):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    m = tf.cast(mask, tf.float32)

    # Forest = pure green (0, 255, 0) in DeepGlobe palette
    forest_mask = tf.logical_and(
        tf.logical_and(m[:,:,1] > 200, m[:,:,0] < 50), m[:,:,2] < 50
    )
    forest_mask = tf.cast(forest_mask, tf.float32)
    forest_mask = tf.expand_dims(forest_mask, -1)

    if augment:
        combined = tf.concat([img, forest_mask], axis=-1)
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        img         = combined[:, :, :3]
        forest_mask = combined[:, :, 3:]
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.clip_by_value(img, 0.0, 1.0)

    return img, forest_mask

# --- MODEL ---
def conv_block(x, filters, dropout_rate=0.0):
    reg = regularizers.l2(1e-4)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x

def build_unet():
    inputs = layers.Input((*IMG_SIZE, 3))

    f1 = conv_block(inputs, 32);         p1 = layers.MaxPooling2D()(f1)
    f2 = conv_block(p1, 64);             p2 = layers.MaxPooling2D()(f2)
    f3 = conv_block(p2, 128);            p3 = layers.MaxPooling2D()(f3)
    f4 = conv_block(p3, 256);            p4 = layers.MaxPooling2D()(f4)

    b  = conv_block(p4, 512, dropout_rate=0.5)

    u1 = layers.UpSampling2D()(b);  u1 = layers.Concatenate()([u1, f4]); d1 = conv_block(u1, 256, dropout_rate=0.3)
    u2 = layers.UpSampling2D()(d1); u2 = layers.Concatenate()([u2, f3]); d2 = conv_block(u2, 128, dropout_rate=0.2)
    u3 = layers.UpSampling2D()(d2); u3 = layers.Concatenate()([u3, f2]); d3 = conv_block(u3, 64)
    u4 = layers.UpSampling2D()(d3); u4 = layers.Concatenate()([u4, f1]); d4 = conv_block(u4, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    return models.Model(inputs, outputs)

# --- WEIGHTED LOSS (handles class imbalance) ---
def make_weighted_bce(pos_weight):
    """Binary cross-entropy where forest pixels are weighted by pos_weight."""
    def weighted_bce(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight_map = y_true * pos_weight + (1.0 - y_true)
        return tf.reduce_mean(weight_map * bce)
    return weighted_bce

# --- SETUP ---
image_paths, mask_paths = load_paths(DATA_DIR)
split = int(0.8 * len(image_paths))

print("Computing class balance from training masks...")
pos_weight = compute_pos_weight(mask_paths[:split], n=100)

train_ds = (tf.data.Dataset.from_tensor_slices((image_paths[:split], mask_paths[:split]))
            .shuffle(1000)
            .map(lambda i, m: process_path(i, m, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((image_paths[split:], mask_paths[split:]))
          .map(lambda i, m: process_path(i, m, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=make_weighted_bce(pos_weight),
    metrics=[
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='forest_iou'),
        tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name='mean_iou'),
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_forest_iou', patience=8,
                                     restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_forest_iou', factor=0.5,
                                         patience=3, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_forest_iou',
                                       save_best_only=True, mode='max'),
]

print(f"\nTraining on {split} images, validating on {len(image_paths)-split}...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- VISUALIZE PREDICTIONS ---
def visualize_predictions(image_paths, mask_paths, model, n=6, start_idx=0):
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    for row, idx in enumerate(range(start_idx, start_idx + n)):
        img_raw = load_img(image_paths[idx], target_size=IMG_SIZE)
        img_arr = img_to_array(img_raw) / 255.0
        pred    = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0]
        binary_pred = (pred > 0.5).squeeze()

        mask = tf.io.read_file(mask_paths[idx])
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
        m    = tf.cast(mask, tf.float32)
        true_mask = tf.logical_and(
            tf.logical_and(m[:,:,1] > 200, m[:,:,0] < 50), m[:,:,2] < 50
        ).numpy()

        forest_pct = np.mean(binary_pred) * 100
        iou = np.sum(binary_pred & true_mask) / np.sum(binary_pred | true_mask) if np.sum(binary_pred | true_mask) > 0 else 0.0

        axes[row, 0].imshow(img_raw);                          axes[row, 0].set_title("Satellite");           axes[row, 0].axis('off')
        axes[row, 1].imshow(true_mask, cmap='Greens');         axes[row, 1].set_title("Ground truth");        axes[row, 1].axis('off')
        axes[row, 2].imshow(binary_pred, cmap='Greens');       axes[row, 2].set_title(f"Pred  Forest:{forest_pct:.0f}%  IoU:{iou:.2f}"); axes[row, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Show 6 validation samples — mix of forested and non-forested
print("\nValidation predictions:")
visualize_predictions(image_paths, mask_paths, model, n=6, start_idx=split)