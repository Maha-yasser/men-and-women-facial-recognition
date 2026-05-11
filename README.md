
# Men & Women Facial Recognition 👤
A binary gender classifier built with EfficientNetB0 and fine-tuning.

## Model Performance
| Metric | Score |
|--------|-------|
| Test Accuracy | 97.20% % |
| Loss | 0.1862 |
| Men F1-Score | 0.97 |
| Women F1-Score | 0.97 |

## Datasets Used for Training
| Dataset | Link |
|---------|------|
| Human Images Dataset (Men and Women) | [Kaggle](https://www.kaggle.com/datasets/playlist/men-women-classification) |
| Men and Women Classification | [Kaggle](https://www.kaggle.com/datasets/playlist/men-women-classification) |

Download via Kaggle API:
```bash
kaggle datasets download -d marquis03/human-images-dataset-men-and-women
kaggle datasets download -d playlist/men-women-classification
```

## Load the Model

### Option 1 — Clone repo (requires Git LFS)
```bash
git lfs install
git clone https://github.com/Maha-yasser/men-and-women-facial-recognition.git
```
```python
import tensorflow as tf
model = tf.keras.models.load_model('men-and-women-facial-recognition/efficientnet_gender.keras')
```

### Option 2 — Direct download
```python
import requests

url = "https://media.githubusercontent.com/media/Maha-yasser/men-and-women-facial-recognition/main/efficientnet_gender.keras"
response = requests.get(url, stream=True)
with open("efficientnet_gender.keras", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

model = tf.keras.models.load_model('efficientnet_gender.keras')
```

### Option 3 — Kaggle Notebook (easiest)
```python
import tensorflow as tf
model = tf.keras.models.load_model('/kaggle/input/efficientnet-gender/efficientnet_gender.keras')
```

## Predict on a New Image
```python
import numpy as np
from PIL import Image

def predict_gender(image_path, model):
    img = Image.open(image_path).resize((224, 224))
    x   = np.array(img) / 255.0
    x   = np.expand_dims(x, axis=0)
    prob = float(model.predict(x)[0][0])
    label = "Women" if prob > 0.5 else "Men"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, round(confidence * 100, 2)

label, confidence = predict_gender("face.jpg", model)
print(f"Prediction: {label} ({confidence}% confident)")
```

## Requirements
```
tensorflow>=2.12
numpy
pillow
```
