import tensorflow as tf
import keras
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array

# --- 1. GLOBAL LOSS FUNCTIONS (Required for Flawless Serialization) ---

@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss: immune to class imbalance, directly optimises overlap."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

@tf.keras.utils.register_keras_serializable(package="Custom")
def bce_dice_loss(y_true, y_pred):
    """Combined BCE + Dice: BCE stabilises early training, Dice handles imbalance."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

# --- 2. DEFORESTATION CLASSIFIER CLASS ---

class DeforestationClassifier:
    def __init__(
        self,
        data_dir,
        img_size=(256, 256),
        batch_size=16,
        epochs=40,
        learning_rate=1e-4,
        save_path='deforestation_model_final.keras',
    ):
        self.data_dir      = data_dir
        self.img_size      = img_size
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.save_path     = save_path
        self._model        = None

    def _load_paths(self):
        base = self.data_dir
        sat_images  = sorted([os.path.join(base, f) for f in os.listdir(base) if f.endswith('_sat.jpg')])
        mask_images = sorted([os.path.join(base, f) for f in os.listdir(base) if f.endswith('_mask.png')])
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(sat_images))
        return [sat_images[i] for i in indices], [mask_images[i] for i in indices]

    def _extract_forest_mask(self, mask_tensor):
        """Logic for DeepGlobe: Forest is approx (0, 255, 0)."""
        forest = tf.logical_and(
            tf.logical_and(mask_tensor[:, :, 1] > 200, mask_tensor[:, :, 0] < 50),
            mask_tensor[:, :, 2] < 50
        )
        return tf.cast(forest, tf.float32)

    def _process_path(self, image_path, mask_path, augment=False):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.resize(mask, self.img_size, method='nearest')
        m = tf.cast(mask, tf.float32)

        forest_mask = tf.expand_dims(self._extract_forest_mask(m), -1)

        if augment:
            combined = tf.concat([img, forest_mask], axis=-1)
            combined = tf.image.random_flip_left_right(combined)
            combined = tf.image.random_flip_up_down(combined)
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            combined = tf.image.rot90(combined, k=k)
            img = combined[:, :, :3]
            forest_mask = combined[:, :, 3:]
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.clip_by_value(img, 0.0, 1.0)

        return img, forest_mask

    @staticmethod
    def _conv_block(x, filters, dropout_rate=0.0):
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

    def _build_unet(self):
        inputs = layers.Input((*self.img_size, 3))
        # Encoder
        f1 = self._conv_block(inputs, 32);  p1 = layers.MaxPooling2D()(f1)
        f2 = self._conv_block(p1,     64);  p2 = layers.MaxPooling2D()(f2)
        f3 = self._conv_block(p2,    128);  p3 = layers.MaxPooling2D()(f3)
        f4 = self._conv_block(p3,    256);  p4 = layers.MaxPooling2D()(f4)
        # Bridge
        b = self._conv_block(p4, 512, dropout_rate=0.5)
        # Decoder
        u1 = layers.Concatenate()([layers.UpSampling2D()(b),  f4]); d1 = self._conv_block(u1, 256, dropout_rate=0.3)
        u2 = layers.Concatenate()([layers.UpSampling2D()(d1), f3]); d2 = self._conv_block(u2, 128, dropout_rate=0.2)
        u3 = layers.Concatenate()([layers.UpSampling2D()(d2), f2]); d3 = self._conv_block(u3,  64)
        u4 = layers.Concatenate()([layers.UpSampling2D()(d3), f1]); d4 = self._conv_block(u4,  32)

        outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
        return models.Model(inputs, outputs)

    def train(self):
        image_paths, mask_paths = self._load_paths()
        split = int(0.8 * len(image_paths))

        train_ds = (tf.data.Dataset.from_tensor_slices((image_paths[:split], mask_paths[:split]))
                    .shuffle(1000)
                    .map(lambda i, m: self._process_path(i, m, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(self.batch_size).prefetch(tf.data.AUTOTUNE))

        val_ds = (tf.data.Dataset.from_tensor_slices((image_paths[split:], mask_paths[split:]))
                  .map(lambda i, m: self._process_path(i, m, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(self.batch_size).prefetch(tf.data.AUTOTUNE))

        self._model = self._build_unet()
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=bce_dice_loss,
            metrics=[
                tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='forest_iou'),
                tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name='mean_iou'),
            ]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_forest_iou', patience=10, restore_best_weights=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_forest_iou', factor=0.5, patience=4, mode='max'),
            tf.keras.callbacks.ModelCheckpoint(self.save_path, monitor='val_forest_iou', save_best_only=True, mode='max'),
        ]

        self._model.fit(train_ds, validation_data=val_ds, epochs=self.epochs, callbacks=callbacks)
        return self._model

    def get(self, new=False):
        if not new and os.path.exists(self.save_path):
            print(f"Loading saved model from {self.save_path}...")
            # Use registered function names in custom_objects
            self._model = tf.keras.models.load_model(
                self.save_path,
                custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss}
            )
        else:
            if os.path.exists(self.save_path): os.remove(self.save_path)
            self.train()
        return self._model

    def predict(self, image, threshold=0.1, normalize=False):
        """
        Args:
            normalize: Stretch each image's per-channel 2nd–98th percentile to [0, 1]
                       before inference. Useful when input images have a different
                       brightness/contrast distribution than the training data (e.g.
                       Landsat GEE composites vs. DeepGlobe DigitalGlobe imagery).
        """
        if self._model is None: raise RuntimeError("Model not loaded.")

        single = not isinstance(image, list)
        images = [image] if single else image
        batch = []
        for img in images:
            if isinstance(img, str):
                arr = img_to_array(load_img(img, target_size=self.img_size)) / 255.0
            else:
                arr = np.array(img, dtype=np.float32)
                if arr.max() > 1.0: arr /= 255.0
                if arr.shape[:2] != self.img_size:
                    arr = tf.image.resize(arr, self.img_size).numpy()
            if normalize:
                # Step 1: per-channel percentile stretch (fixes brightness mismatch)
                p2, p98 = np.percentile(arr, (2, 98), axis=(0, 1), keepdims=True)
                rng = p98 - p2
                rng[rng == 0] = 1.0
                arr = np.clip((arr - p2) / rng, 0.0, 1.0)
                # Step 2: amplify relative greenness so forest (G > R) reads as
                # vivid green, matching the DeepGlobe color signature the model
                # was trained on. Non-vegetated areas (R >= G) are unchanged.
                g, r = arr[:, :, 1], arr[:, :, 0]
                veg = np.clip((g - r) / (g + r + 1e-6), 0.0, 1.0)
                arr = arr.copy()
                arr[:, :, 1] = np.clip(arr[:, :, 1] + veg * 0.4, 0.0, 1.0)
                arr[:, :, 0] = np.clip(arr[:, :, 0] - veg * 0.2, 0.0, 1.0)
            batch.append(arr)

        preds = self._model.predict(np.stack(batch), verbose=0)
        masks = (preds.squeeze(-1) > threshold)
        return masks[0] if single else masks

    def visualize_predictions(self, n_each=3):
        image_paths, mask_paths = self._load_paths()
        split = int(0.8 * len(image_paths))
        val_imgs, val_masks = image_paths[split:], mask_paths[split:]

        forested, non_forested = [], []
        for idx in range(min(300, len(val_imgs))):
            mask = tf.io.read_file(val_masks[idx])
            mask = tf.image.decode_png(mask, channels=3)
            mask = tf.image.resize(mask, self.img_size, method='nearest')
            pct = tf.reduce_mean(self._extract_forest_mask(tf.cast(mask, tf.float32))).numpy() * 100
            if pct > 10 and len(forested) < n_each: forested.append(idx)
            elif pct < 5 and len(non_forested) < n_each: non_forested.append(idx)
            if len(forested) == n_each and len(non_forested) == n_each: break

        samples = forested + non_forested
        fig, axes = plt.subplots(len(samples), 3, figsize=(4, 2 * len(samples)))
        for row, idx in enumerate(samples):
            img_raw = load_img(val_imgs[idx], target_size=self.img_size)
            binary_pred = self.predict(val_imgs[idx])
            
            mask_raw = tf.io.read_file(val_masks[idx])
            mask_raw = tf.image.decode_png(mask_raw, channels=3)
            mask_raw = tf.image.resize(mask_raw, self.img_size, method='nearest')
            true_mask = self._extract_forest_mask(tf.cast(mask_raw, tf.float32)).numpy().astype(bool)

            true_deforestation_pct = (1 - true_mask.mean()) * 100
            pred_deforestation_pct = (1 - binary_pred.mean()) * 100

            axes[row][0].imshow(img_raw); axes[row][0].set_title("Satellite"); axes[row][0].axis('off')
            axes[row][1].imshow(true_mask, cmap='gray'); axes[row][1].set_title(f"Ground Truth\nDeforestation: {true_deforestation_pct:.1f}%"); axes[row][1].axis('off')
            axes[row][2].imshow(binary_pred, cmap='Greens_r'); axes[row][2].set_title(f"Prediction\nDeforestation: {pred_deforestation_pct:.1f}%"); axes[row][2].axis('off')
        plt.tight_layout(); plt.show()

# --- RUN ---
if __name__ == '__main__':
    DATA_DIR = '/Users/jsmith/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2' + '/train'
    classifier = DeforestationClassifier(data_dir=DATA_DIR)
    classifier.get() 
    classifier.visualize_predictions(n_each=5)