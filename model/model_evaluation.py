import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os

# Debug: Check if files exist
print("Files in datasets folder:", os.listdir('./datasets'))

# Load data
X_test = np.load('./datasets/X_train.npy')  # Replace with your actual path
y_test = np.load('./datasets/y_train.npy')

print(f"X_test shape: {X_test.shape}")  # Should be (num_samples, height, width, channels)
print(f"y_test shape: {y_test.shape}")  # Should be (num_samples,)

# Exit if data is empty
if len(X_test) == 0:
    raise ValueError("X_test is empty! Check your .npy files.")