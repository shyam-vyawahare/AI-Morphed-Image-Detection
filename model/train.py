# -------------------------------
# Forensic Morphing Detection Training Script
# -------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from skimage.filters import gaussian

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import EfficientNetB4 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore

# ----------------------
# JPEG Artifact Simulation
# ----------------------

def jpeg_artifacts_numpy(image):
    image = (image * 255).astype(np.uint8)
    quality = np.random.randint(40, 90)
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded.astype(np.float32) / 255.0

def preprocessing_wrapper(image):
    image = tf.numpy_function(jpeg_artifacts_numpy, [image], tf.float32)
    image.set_shape([None, None, 3])
    return image

# ----------------------
# Build Model
# ----------------------

def build_model():
    base_model = EfficientNetB4(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:150]:
        layer.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

# ----------------------
# Focal Loss Function
# ----------------------

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss_fn

# ----------------------
# Constants
# ----------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
TRAIN_PATH = './datasets/train'

# ----------------------
# Data Generators
# ----------------------

train_gen = ImageDataGenerator(
    preprocessing_function=preprocessing_wrapper,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    validation_split=0.2
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42,
    classes=['morphed', 'original']
)

val_data = val_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    classes=['morphed', 'original']
)

# ----------------------
# Class Weights
# ----------------------

class_names = ['morphed', 'original']
counts_by_index = {
    0: len(os.listdir(os.path.join(TRAIN_PATH, 'morphed'))),
    1: len(os.listdir(os.path.join(TRAIN_PATH, 'original')))
}
total = sum(counts_by_index.values())
class_weights = {idx: total / (len(counts_by_index) * count) for idx, count in counts_by_index.items()}
print(f"Class weights: {class_weights}")

# ----------------------
# Compile Model
# ----------------------

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# ----------------------
# Callbacks
# ----------------------

callbacks = [
    EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, mode='max'),
    ModelCheckpoint('./saved_model/best_model.keras', monitor='val_auc', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.2)
]

# ----------------------
# Train
# ----------------------

history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    validation_data=val_data,
    validation_steps=val_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ----------------------
# Evaluation
# ----------------------

print("\nFinal Evaluation:")
val_images, val_labels = next(val_data)
val_preds = (model.predict(val_images) > 0.5).astype(int)

print(classification_report(val_labels, val_preds, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(val_labels, val_preds))

# ----------------------
# Save Model
# ----------------------

model.save('./saved_model/morph_detection_model.keras')
print("Model saved successfully.")

# ----------------------
# Visualization
# ----------------------

plt.figure(figsize=(15, 5))
for i, metric in enumerate(['accuracy', 'loss', 'auc']):
    plt.subplot(1, 3, i+1)
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history[f'val_{metric}'], label='Val')
    plt.title(metric.capitalize())
    plt.legend()
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()
