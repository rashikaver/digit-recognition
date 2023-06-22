import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Step 1: Load and preprocess the data
df = pd.read_csv('C:\\Users\\Rashika\\Desktop\\codeclause.pyc\\train.csv')

# Split the dataset into features (pixels) and labels (digits)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Normalize the pixel values between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Evaluate the model
_, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

predictions = np.argmax(model.predict(X_test), axis=-1)
print(classification_report(y_test, predictions))

# Step 5: Make predictions
image_index = 10 # Change this to test different images

plt.imshow(X_test[image_index].reshape(28, 28), cmap='gray')
plt.show()

prediction = np.argmax(model.predict(X_test[image_index].reshape(1, -1)))
print("Predicted digit:", prediction)
