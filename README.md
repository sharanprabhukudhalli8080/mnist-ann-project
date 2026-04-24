# 🧠 Handwritten Digit Recognition using ANN

## 📌 Overview

This project builds an Artificial Neural Network (ANN) to classify handwritten digits (0–9) using the MNIST dataset (CSV format).

## 📂 Project Structure

```
mnist-ann-project/
│── src/            # Training & prediction scripts
│── models/         # Saved trained model
│── notebook/       # Colab notebook
│── README.md
```

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* Pandas, NumPy
* Matplotlib

## 🚀 How to Run

```bash
pip install -r requirements.txt
cd src
python train.py
python predict.py
```

## 📊 Model Details

* Input: 28x28 pixel images
* Hidden Layers: 128, 64 neurons
* Output: 10 classes (digits 0–9)

## 📈 Accuracy

Achieved ~97% accuracy on test dataset

## 💡 Features

* Manual dataset handling (CSV)
* Data preprocessing & normalization
* ANN model training
* Prediction visualization

## 🔥 Future Improvements

* Upgrade to CNN (99% accuracy)
* Deploy using Streamlit
* Custom handwritten dataset

