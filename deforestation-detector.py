import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

# --- CONFIGURATION ---
IMG_SIZE        = (256, 256)
BATCH_SIZE      = 16
EPOCHS          = 40
LEARNING_RATE   = 1e-4
MODEL_SAVE_PATH = 'deforestation_model_final.keras'
DATA_DIR        = '/kaggle/input/deepglobe-land-cover-classification-dataset/train'

if os.path.exists(MODEL_SAVE_PATH):
    os.remove(MODEL_SAVE_PATH)
    print("Old model removed. Starting fresh.")

# --- PATH LOADER ---
def load_paths(base_path):
    sat_images  = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('_sat.jpg')])
    mask_images = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('_mask.png')])
    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(sat_images))
    return [sat_images[i] for i in indices], [mask_images[i] for i in indices]

# --- FOREST MASK EXTRACTOR ---
def extract_forest_mask(mask_tensor):
    """mask_tensor: float32 HxWx3. Returns float32 HxW with 1=forest, 0=other."""
    forest = tf.logical_and(
        tf.logical_and(mask_tensor[:,:,1] > 200, mask_tensor[:,:,0] < 50),
        mask_tensor[:,:,2] < 50
    )
    return tf.cast(forest, tf.float32)

# --- DATA PIPELINE ---
def process_path(image_path, mask_path, augment=False):
    img  = tf.io.read_file(image_path)
    img  = tf.image.decode_jpeg(img, channels=3)
    img  = tf.image.resize(img, IMG_SIZE) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    m    = tf.cast(mask, tf.float32)

    forest_mask = tf.expand_dims(extract_forest_mask(m), -1)

    if augment:
        combined    = tf.concat([img, forest_mask], axis=-1)
        combined    = tf.image.random_flip_left_right(combined)
        combined    = tf.image.random_flip_up_down(combined)
        k           = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        combined    = tf.image.rot90(combined, k=k)
        img         = combined[:, :, :3]
        forest_mask = combined[:, :, 3:]
        img         = tf.image.random_contrast(img, 0.8, 1.2)
        img         = tf.image.random_brightness(img, 0.1)
        img         = tf.clip_by_value(img, 0.0, 1.0)

    return img, forest_mask

# --- LOSSES ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss: immune to class imbalance, directly optimises overlap."""
    y_true_f    = tf.reshape(y_true, [-1])
    y_pred_f    = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def bce_dice_loss(y_true, y_pred):
    """Combined BCE + Dice: BCE stabilises early training, Dice handles imbalance."""
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

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

    f1 = conv_block(inputs, 32);  p1 = layers.MaxPooling2D()(f1)
    f2 = conv_block(p1,     64);  p2 = layers.MaxPooling2D()(f2)
    f3 = conv_block(p2,    128);  p3 = layers.MaxPooling2D()(f3)
    f4 = conv_block(p3,    256);  p4 = layers.MaxPooling2D()(f4)

    b = conv_block(p4, 512, dropout_rate=0.5)

    u1 = layers.Concatenate()([layers.UpSampling2D()(b),  f4]); d1 = conv_block(u1, 256, dropout_rate=0.3)
    u2 = layers.Concatenate()([layers.UpSampling2D()(d1), f3]); d2 = conv_block(u2, 128, dropout_rate=0.2)
    u3 = layers.Concatenate()([layers.UpSampling2D()(d2), f2]); d3 = conv_block(u3,  64)
    u4 = layers.Concatenate()([layers.UpSampling2D()(d3), f1]); d4 = conv_block(u4,  32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    return models.Model(inputs, outputs)

# --- SETUP ---
image_paths, mask_paths = load_paths(DATA_DIR)
split = int(0.8 * len(image_paths))

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
    loss=bce_dice_loss,
    metrics=[
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='forest_iou'),
        tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name='mean_iou'),
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_forest_iou', patience=10,
                                     restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_forest_iou', factor=0.5,
                                         patience=4, mode='max', min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_forest_iou',
                                       save_best_only=True, mode='max'),
]

print(f"Training on {split} images, validating on {len(image_paths) - split}...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- VISUALIZE: balanced forested / non-forested sample ---
def visualize_predictions(image_paths, mask_paths, model, n_each=3, start_idx=0):
    val_imgs  = image_paths[start_idx:]
    val_masks = mask_paths[start_idx:]

    forested, non_forested = [], []
    for idx in range(min(300, len(val_imgs))):
        mask = tf.io.read_file(val_masks[idx])
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
        pct  = tf.reduce_mean(extract_forest_mask(tf.cast(mask, tf.float32))).numpy() * 100
        if pct > 10 and len(forested) < n_each:
            forested.append(idx)
        elif pct < 5 and len(non_forested) < n_each:
            non_forested.append(idx)
        if len(forested) == n_each and len(non_forested) == n_each:
            break

    samples = forested + non_forested
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for row, idx in enumerate(samples):
        img_raw     = load_img(val_imgs[idx], target_size=IMG_SIZE)
        img_arr     = img_to_array(img_raw) / 255.0
        pred        = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0]
        binary_pred = (pred > 0.5).squeeze()

        mask      = tf.io.read_file(val_masks[idx])
        mask      = tf.image.decode_png(mask, channels=3)
        mask      = tf.image.resize(mask, IMG_SIZE, method='nearest')
        true_mask = extract_forest_mask(tf.cast(mask, tf.float32)).numpy().astype(bool)

        forest_pct = np.mean(binary_pred) * 100
        true_pct   = np.mean(true_mask) * 100
        union      = np.sum(binary_pred | true_mask)
        iou        = np.sum(binary_pred & true_mask) / union if union > 0 else 1.0

        axes[row][0].imshow(img_raw)
        axes[row][0].set_title(f"Satellite  (true forest: {true_pct:.0f}%)")
        axes[row][0].axis('off')

        axes[row][1].imshow(true_mask.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
        axes[row][1].set_title("Ground truth")
        axes[row][1].axis('off')

        axes[row][2].imshow(binary_pred.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
        axes[row][2].set_title(f"Pred  Forest:{forest_pct:.0f}%  IoU:{iou:.2f}")
        axes[row][2].axis('off')

    plt.tight_layout()
    plt.show()

print("\nValidation predictions (3 forested + 3 non-forested):")
visualize_predictions(image_paths, mask_paths, model, n_each=3, start_idx=split)