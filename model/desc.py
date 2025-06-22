from transformers import BlipProcessor, BlipForConditionalGeneration # type: ignore
from PIL import Image
import requests

# Load model and processor
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# Load image
image = Image.open(requests.get('./static/images/skv.jpg', stream=True).raw)

# Prepare inputs
inputs = processor(image, return_tensors='pt')

# Generate caption
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated description:", caption)