# 📘 Smart Digit Recognizer

Digit recognition using multiple Machine Learning models on the MNIST dataset, with training, evaluation, and performance comparison.  
Additionally, a Convolutional Neural Network (CNN) model 🖼️ was trained for improved accuracy, and a Streamlit-based web interface allows users to draw digits and get predictions in real-time.  

🔗 **Try the Streamlit App here:** [Smart Digit Recognizer App](YOUR_STREAMLIT_LINK_HERE)

---

## 📂 Dataset

- **Source:** [MNIST 60,000 Handwritten Number Images (Train ) (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
- **Used:** Training split (60,000 samples)  
- **Image size:** 28 × 28 pixels (flattened into 784 features)  
- **Target:** Digits 0–9  

---

## ⚙️ Workflow

1. **Data Loading & Preparation**
   - Loaded dataset into a DataFrame.
   - Separated features (pixels) and target labels.
   - Renamed columns for clarity.

2. **Exploratory Data Analysis (EDA)**
   - Checked class distribution (balanced dataset).
   - Visualized some sample digits.

3. **Preprocessing**
   - Normalized pixel values (0–255 → 0–1).
   - Reshaped images to (28,28,1) for CNN input.
   - Split into training (80%) and validation (20%).

4. **Model Training**
   - Trained and compared five classical ML models:
     - 🧠 Multi-layer Perceptron (MLP)  
     - 🔤 Perceptron  
     - 🌳 Decision Tree  
     - 🌲 Random Forest  
     - ⚡ XGBoost  
   - 🖼️ Convolutional Neural Network (CNN) trained on reshaped 28×28 images with data augmentation, achieving higher accuracy than classical ML models.

5. **Evaluation**
   - Measured Accuracy, Precision, Recall, F1-score.
   - Compared results across all models.
   - **Best Machine Learning model:** MLPClassifier among the classical ML models.  
   - **Best overall model:** CNN, outperforming all other models in accuracy and metrics.

---

## 📊 Results

| Model                   | Accuracy | Precision | Recall | F1-score |
|-------------------------|----------|-----------|--------|----------|
| 🧠 MLPClassifier        | 0.976    | 0.97      | 0.97   | 0.97     |
| 🔤 Perceptron           | 0.920    | 0.92      | 0.92   | 0.92     |
| 🌳 Decision Tree        | 0.860    | 0.86      | 0.86   | 0.86     |
| 🌲 Random Forest        | 0.960    | 0.96      | 0.96   | 0.96     |
| ⚡ XGBoost              | 0.968    | 0.97      | 0.97   | 0.97     |
| 🖼️ CNN (Deep Learning) | 0.990    | 0.99      | 0.99   | 0.99     |

> ✅ **Best Machine Learning Model:** MLPClassifier – highest performance among classical ML models.  
> ✅ **Best Overall Model:** CNN – achieved the highest performance across all metrics.

---

## 🖥️ Deployment

- The CNN model was deployed using **Streamlit**, providing an interactive web interface.  
- Users can **draw a digit (0–9) on the canvas** and click "Predict" to see the model's prediction in real-time.  
- Features:
  - Freehand drawing canvas.
  - Instant prediction with emoji representation.
  - Clean, modern UI with custom styling.  
- This deployment demonstrates a **real-world application of the model**, bridging ML training and user interaction.

---

## 🚀 Conclusion

- This project shows the impact of different ML algorithms on digit recognition.  
- Simple models (Perceptron, Decision Tree) gave decent results but lacked generalization.  
- Ensemble methods (Random Forest, XGBoost) performed strongly.  
- **Best Machine Learning model:** MLPClassifier among classical models.  
- **Best overall model:** CNN, providing the **highest accuracy** and a **user-friendly interface** for real-time digit recognition via Streamlit.