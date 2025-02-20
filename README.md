# Iris Classification Using Neural Network

This project builds a **Neural Network (NN) model** to classify the **Iris dataset** into three categories: **Setosa, Versicolor, and Virginica**. The model is built using **TensorFlow/Keras**.

---

## 📌 **Project Steps**

1. **Import Necessary Libraries** - Load TensorFlow, NumPy, Matplotlib, and Scikit-learn.
2. **Load and Preprocess Data** - Load the Iris dataset, normalize features, and one-hot encode labels.
3. **Split Data** - 80% for training, 20% for testing.
4. **Build the Neural Network Model** - A simple feedforward neural network with three layers.
5. **Compile the Model** - Define optimizer, loss function, and evaluation metric.
6. **Train the Model** - Fit the model on training data.
7. **Evaluate the Model** - Test on unseen data and generate performance metrics.
8. **Save and Load Model** - Store the trained model for future use.
9. **Visualize Model and Classification Report** - Plot model architecture and generate a classification report.

---

## 📂 **Dataset**
- The **Iris dataset** consists of 150 samples with **4 numerical features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- There are **3 classes (Setosa, Versicolor, Virginica)** with **50 samples each**.

---

## ⚙️ **Requirements**
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## 🛠 **How to Run the Project**

### **1️⃣ Load and Preprocess Data**
- Load the Iris dataset.
- Normalize features (feature scaling: values between 0 and 1).
- Convert class labels into one-hot encoded format.

### **2️⃣ Split Data**
- Use an 80-20 split for training and testing.

### **3️⃣ Build the Neural Network**
- **Input Layer:** 4 neurons (one for each feature).
- **Hidden Layer:** 10 neurons (ReLU activation).
- **Output Layer:** 3 neurons (softmax activation for classification).

### **4️⃣ Compile & Train the Model**
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- Train the model for a defined number of epochs.

### **5️⃣ Evaluate the Model**
- Measure **accuracy** and **loss**.
- Generate a **classification report** with precision, recall, and F1-score.

### **6️⃣ Save and Load the Model**
- Save the trained model (`.h5` format) to reuse without retraining.
- Load the saved model and verify predictions.

### **7️⃣ Visualize Model and Classification Report**
- **Plot model architecture** using `plot_model()`.
- Generate a **classification report** using Scikit-learn.

---

## 📊 **Results & Output**
A sample **classification report**:
```
Classification Report:
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00        10
  Versicolor       1.00      0.90      0.95        10
   Virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

---

## 🚀 **Next Steps**
1. **Improve Model Performance** - Tune hyperparameters (hidden layers, neurons, batch size).
2. **Convert to TensorFlow Lite** - Deploy the model on mobile devices.
3. **Deploy as API** - Use Flask or FastAPI to serve the model.

---

## 📜 **License**
This project is open-source and free to use.

---

## 💡 **Author**
Developed as part of an **Iris Classification Neural Network** project using TensorFlow & Keras.

