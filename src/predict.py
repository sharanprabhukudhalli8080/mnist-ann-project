import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("../models/mnist_model.h5")

# Load test data
test_data = pd.read_csv("mnist_test.csv")

x_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

x_test = x_test.reshape(-1, 28, 28)

# Pick sample
index = 20
image = x_test[index]

# Predict
prediction = model.predict(image.reshape(1,28,28))
digit = np.argmax(prediction)

# Show result
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {digit}, Actual: {y_test[index]}")
plt.show()
