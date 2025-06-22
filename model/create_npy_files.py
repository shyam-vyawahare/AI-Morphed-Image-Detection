import numpy as np
import os
from PIL import Image

def images_to_npy(image_folder, label):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg','.webp')):
            try:
                img = Image.open(os.path.join(image_folder, filename))
                img = img.resize((224, 224))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img) / 255.0
                images.append(img_array)
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
    return np.array(images), np.full(len(images), label)

# Paths
train_original_path = './datasets/train/original'
train_morphed_path = './datasets/train/morphed'
test_original_path = './datasets/test/original'
test_morphed_path = './datasets/test/morphed'

# Process data
X_train_original, y_train_original = images_to_npy(train_original_path, label=0)
X_train_morphed, y_train_morphed = images_to_npy(train_morphed_path, label=1)

# Debug shapes
print(f"Original images shape: {X_train_original.shape}")  # e.g., (100, 224, 224, 3)
print(f"Morphed images shape: {X_train_morphed.shape}")    # Must match original

# Ensure arrays are 4D before concatenation
assert X_train_original.ndim == 4 and X_train_morphed.ndim == 4, "Arrays must be 4D!"

# Concatenate and save
X_train = np.concatenate([X_train_original, X_train_morphed])
y_train = np.concatenate([y_train_original, y_train_morphed])
np.save('./datasets/X_train.npy', X_train)
np.save('./datasets/y_train.npy', y_train)

print(f"Saved {X_train.shape[0]} training samples.")