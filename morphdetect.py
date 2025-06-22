from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("my_model.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(f"Prediction: {'Morphed' if pred[0][0] > 0.5 else 'Original'} | Confidence: {pred[0][0]:.2f}")

predict_image("C:/Users/ashis/Downloads/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_fake/mid_6_1111.jpg")
predict_image("C:/Users/ashis/Downloads/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00992.jpg")
