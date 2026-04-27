# 🔍 CIFAR-10 AI Image Classifier

> **FSE560 — Foundations of AI | Term Project | Topic T3**  
> A complete end-to-end image classification system using a CNN model and a Streamlit web aggregator.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-orange)](https://huggingface.co/spaces/im17/cifar10-image-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-simple--image--classifier-blue)](https://github.com/imello17/simple-image-classifier)

---

## 📋 Project Overview

This project builds a two-component AI image classification system trained on the **CIFAR-10** dataset:

| Component | Description |
|---|---|
| **AI Model** | Convolutional Neural Network (CNN) trained on 50,000 images across 10 categories |
| **AI Aggregator** | Streamlit web app — upload any photo and get a real-time prediction |

### 🏷️ The 10 Categories
✈️ Airplane &nbsp;|&nbsp; 🚗 Automobile &nbsp;|&nbsp; 🐦 Bird &nbsp;|&nbsp; 🐱 Cat &nbsp;|&nbsp; 🦌 Deer &nbsp;|&nbsp; 🐶 Dog &nbsp;|&nbsp; 🐸 Frog &nbsp;|&nbsp; 🐴 Horse &nbsp;|&nbsp; 🚢 Ship &nbsp;|&nbsp; 🚛 Truck

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **72.47%** |
| Precision (macro) | **72.70%** |
| Recall (macro) | **72.47%** |
| Training images | 50,000 |
| Test images | 10,000 |
| Model parameters | 122,570 |

### Best performing classes
- 🚗 Automobile — 87% precision
- 🚢 Ship — 84% precision

### Most challenging classes
- 🐱 Cat — 52% precision (often confused with dog/deer)
- 🐦 Bird — 60% precision (confused with airplane at 32×32)

---

## 🏗️ Model Architecture

```
Input (32×32×3)
    ↓
Conv2D(32, 3×3, ReLU)     →  896 params
MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU)     →  18,496 params
MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU)     →  36,928 params
    ↓
Flatten → Dense(64, ReLU) →  65,600 params
Dropout(0.5) ← Improvement #1
    ↓
Dense(10, logits)          →  650 params
                              ──────────
                              122,570 total
```

### Key Improvements Over Baseline
1. **Dropout(0.5)** — reduces overfitting by randomly zeroing 50% of neurons during training
2. **EarlyStopping** — monitors val_accuracy with patience=3, restored best weights from Epoch 17

---

## 🚀 Live Demo

Try the app live on Hugging Face Spaces:  
👉 **[https://huggingface.co/spaces/im17/cifar10-image-classifier](https://huggingface.co/spaces/im17/cifar10-image-classifier)**

Upload any photo and the CNN will classify it into one of the 10 CIFAR-10 categories with a confidence score and timing breakdown.

---

## 📁 Project Structure

```
cifar10-project/
├── notebooks/
│   └── Image_recognizer_small_improved.ipynb   # CNN training notebook
├── app/
│   └── app.py                                  # Streamlit aggregator
└── cifar10_cnn_model.h5                        # Trained model weights
```

---

## 🛠️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/imello17/simple-image-classifier.git
cd simple-image-classifier
```

### 2. Install dependencies
```bash
pip install tensorflow streamlit numpy pandas pillow scikit-learn
```

### 3. Train the model (optional — skip if .h5 already exists)
Open and run `notebooks/Image_recognizer_small_improved.ipynb` in VS Code or Jupyter.

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```

Then open your browser at `http://localhost:8501`

---

## 📦 Dependencies

```
tensorflow >= 2.20
streamlit
numpy
pandas
pillow
scikit-learn
seaborn
matplotlib
```

---

## 📈 Performance Comparison: Model vs Aggregator

| Metric | CNN Model (batch) | Streamlit Aggregator |
|---|---|---|
| Accuracy | 72.47% | ~72.47% (same weights) |
| Per-image latency | 0.66 ms | ~470 ms |
| Speed ratio | Baseline | ~712x slower |
| Interface | None | Browser UI |

The 712x latency difference is due to single-image preprocessing, PIL overhead, and Streamlit framework costs — not model quality.

---

## 🔬 Dataset

**CIFAR-10** (Canadian Institute For Advanced Research)
- 60,000 color images at 32×32 pixels
- 10 balanced classes (6,000 per class)
- 50,000 training / 10,000 test split
- Built into TensorFlow: `keras.datasets.cifar10`

---

## 📄 License

MIT License — free to use and modify.

---

## 👤 Author

**[Your Name]** — FSE560, Arizona State University, April 2026
