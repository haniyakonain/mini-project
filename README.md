# Multiple Types of Cancer - Mini Project

## 🔬 Project Overview

Cancer is the second leading cause of death worldwide. Early detection is critical for improving patient outcomes. This project presents a deep learning-based approach for classifying CT/MRI images into multiple cancer types using advanced Convolutional Neural Network (CNN) models. The key innovation lies in combining **Transfer Learning**, **Bayesian Hyperparameter Optimization**, and **Learning Without Forgetting (LwF)** to achieve high accuracy in multi-type cancer prediction.

## 📁 Cancer Types Covered

* Lung Cancer
* Brain Tumor
* Breast Cancer
* Cervical Cancer
  *(Total: 8 types as per dataset)*

---

## 🧠 Models Used

* **VGG19** (Primary architecture)
* MobileNet (for comparison)
* DenseNet (for comparison)
* CNN (custom variants)

---

## 🔧 Techniques Applied

* **Transfer Learning** using ImageNet-pretrained models
* **Bayesian Optimization** for hyperparameter tuning
* **Learning Without Forgetting (LwF)** to avoid catastrophic forgetting in transfer learning
* Extensive image preprocessing and augmentation
* Accuracy & loss tracking with graphical visualization

---

## 📊 Results

| Dataset Split | Accuracy |
| ------------- | -------- |
| Training Set  | \~97.2%  |
| Test Set      | \~94.02% |

---

## 🛠 Installation & Setup

### ✅ Requirements

* Python 3.6+
* `TensorFlow`, `Keras`, `scikit-learn`, `matplotlib`, `numpy`, `Pillow`
* Recommended IDE: **Spyder3**

### 📦 Installation

```bash
git clone https://github.com/haniyakonain/mini-project.git
cd mini-project/multiple_types_of_cancer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 📁 Project layout

```
mini-project/
└── multiple_types_of_cancer/   # the whole app - no more nested CODE/ folder
    ├── app.py                  # Flask app (routes: /, /index, /dataset, /predicted)
    ├── model/                  # trained .h5 checkpoints + training notebook
    ├── dataset/                # train/test images, shown live on the /dataset page
    ├── static/                 # css, js, uploaded-image previews
    ├── templates/              # base.html + main/index/dataset/result pages
    └── upload/                 # raw uploaded scans
```

### 📁 Dataset

* The shipped `dataset/train` and `dataset/test` folders already contain the four
  cancer classes used to train the model - browse them live on the site's **Dataset**
  page instead of digging through the filesystem.
* To retrain from scratch, the full source dataset is on
  [Kaggle - Multi-Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer).

---

## 🚀 How to Run

1. Preprocess the data and split into training/testing sets.
2. Train the model using `model/main_alz.ipynb`.
3. Evaluate accuracy and visualize graphs.
4. Save the trained model as `.h5`.
5. From `multiple_types_of_cancer/`, run `python app.py` and open `http://127.0.0.1:5000`
   to upload CT/MRI images for real-time classification.

---

## 🖥 System Requirements

### Hardware:

* Processor: Intel i3/i5
* RAM: 4GB+
* Disk: 250GB+

### Software:

* OS: Windows 7/10 or Linux
* IDE: Spyder / Jupyter Notebook

---

## 📌 Features

* Multi-cancer classification
* Pretrained CNN integration
* Real-time predictions
* User-friendly GUI (HTML templates)
* Scalable architecture for new cancer types

---

## 📸 Sample Screenshots

> Include screenshots of the UI, prediction results, and graphs here

---

## 📈 Future Enhancements

* Real-time deployment via web APIs
* Integration with hospital radiology systems
* Expand to multimodal datasets (e.g., pathology + genetic data)
* Explainability via Grad-CAM

---

## 🤝 Contributions

Pull requests are welcome! Please fork the repo and open a PR with detailed explanation.

---
