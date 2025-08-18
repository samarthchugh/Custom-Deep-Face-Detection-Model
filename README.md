# 🎯 Deep Face Detection & Tracking

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)

This project implements a **deep learning-based face tracker** using **TensorFlow (VGG16 backbone)**,  
**custom training loops**, and **OpenCV** for real-time webcam inference.  

It performs:
- ✅ **Binary Face Detection**  
- ✅ **Bounding Box Regression**  

The dataset was labeled using **[LabelMe](https://github.com/wkentaro/labelme)** and augmented with **[Albumentations](https://albumentations.ai/)**.

---

## 🚀 Features
- 🔹 **Custom training & evaluation loops** (not just `model.fit`)
- 🔹 **Two-branch model**:
  - **Classification** → Detects if a face exists (`BinaryCrossEntropy`)
  - **Regression** → Predicts bounding box coordinates (`custom localization loss`)
- 🔹 **VGG16 Backbone** with `include_top=False`
- 🔹 **Efficient Data Pipeline** using `tf.data` (shuffle, batch, prefetch)
- 🔹 **Real-Time Inference** from webcam using OpenCV
- 🔹 **Augmented Data Support** with Albumentations
- 🔹 **Model Saving/Loading** in `.keras` format

---

## 📂 Project Structure
```bash
Deep-Face-Detection-Model/
│── aug_data/                 # Augmented dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
│── data/                     # Original dataset
│   ├── train/
│   ├── val/
│   └── test/
│
│── facetracker_full.keras     # Saved trained model
│── Deep_Face_Detection_Model.ipynb   # Jupyter notebook (training + inference)
│── requirements.txt           # Dependencies
│── README.md                  # Documentation
│── aug_data.zip               # Zipped augmented dataset
│── venv/                      # Virtual environment (ignored in GitHub)
```

---

## 📦 Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/samarthchugh/deep-face-detection.git
   cd deep-face-detection
   ```

### 2. Create Virtual Environments (recommended):
```bash
conda create -p venv python==3.12 -y
conda activate venv/
```

### 3. Install Dependencies:
```bash 
pip install -r requirements.txt
```

---

## ⚙️ Requirements
```bash
labelme
tensorflow
opencv-python
matplotlib
albumentations
numpy
```

---

## 🏗️ Model Architecture

The model has two outputs:
1. **Face Classification (`sigmoid`)**
2. **Bounding Box Regression (`sigmoid` for [x1,y1,x2,y2])**
```python
def build_model(): 
    input_layer = Input(shape=(120,120,3))
    vgg = VGG16(include_top=False)(input_layer)

    # Classification branch
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding box regression branch
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker
```

---

## 📊 Data Pipeline
Using TensorFlow’s `tf.data` API for efficient loading:
```python
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000).batch(8).prefetch(tf.data.AUTOTUNE)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300).batch(8).prefetch(tf.data.AUTOTUNE)
```

---

### 🧮 Loss Functions
- **Classification Loss**: `BinaryCrossEntropy`
- **Localization Loss**: Custom bounding box regression loss
``` python
classification_loss = tf.keras.losses.BinaryCrossentropy()
regression_loss = localization_loss
```

---

### ⚡ Usage
All workflows (training, evaluation, and inference) are handled in the Jupyter Notebook:
``` bash
jupyter notebook Deep_Face_Detection_Model.ipynb
```
Inside the notebook, you can:
- 🏋️ Train the model with a custom loop (train_step / test_step)
- 🔎 Evaluate the model on validation/test datasets
- 🎥 Run real-time inference with your webcam (press q to quit)

---

### 💾 Saving & Loading Model
```python
# Save locally
facetracker.save("facetracker_full.keras")

# Load loaclly
from tensorflow import keras
model = keras.models.load_model("facetracker_full.keras")
```

---

### 📥 Load Model from Hugging Face Hub
You can also load the model directly from the Hugging Face Hub without downloading manually:
```python
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Download from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="samarthchugh/deep-face-detection",  # change if repo name differs
    filename="facetracker_full.keras"
)

# Load into TensorFlow
model = tf.keras.models.load_model(model_path)
```

---

# 👨‍💻 Author
### Samarth Chugh
- 📧[samarthchugh049@gmail.com](samarthchugh049@gmail.com)
- [Samarth Chugh](www.linkedin.com/in/-samarthchugh)
