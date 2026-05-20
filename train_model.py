import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- SETTINGS ----------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# ---------------- DATASET PATHS ----------------
train_dir = "dataset/train"
val_dir = "dataset/val"

# ---------------- IMAGE GENERATORS ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ---------------- CNN MODEL ----------------
model = models.Sequential([

    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# ---------------- COMPILE MODEL ----------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN MODEL ----------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------------- CREATE MODELS FOLDER ----------------
os.makedirs("models", exist_ok=True)

# ---------------- SAVE MODEL ----------------
model.save("models/fire_detection_model.h5")

print("✅ Model saved successfully inside models folder!")