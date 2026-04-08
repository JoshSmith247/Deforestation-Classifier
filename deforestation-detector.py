import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
DATA_DIR = '/Users/jsmith/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2/train'

print("Loading Data...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Defining the model...")

model = models.Sequential([
    # Rescale pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Convolutional layers to extract spatial features (roads, clearings)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening to feed into the dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting
    
    layers.Dense(1, activation='sigmoid')
])

print("Compiling...")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Standard for yes/no classification
    metrics=['accuracy']
)

model.summary()

# history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# INFERENCE (How to get "Percent Confidence")
# To predict a new image:
# img = tf.keras.utils.load_img('test_image.jpg', target_size=IMG_SIZE)
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
# confidence = model.predict(img_array)[0][0]
# print(f"Deforestation Confidence: {confidence * 100:.2f}%")