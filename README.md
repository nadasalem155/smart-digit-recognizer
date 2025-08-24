# 📘 Smart Digit Recognizer

Handwritten digit recognition using multiple Machine Learning models on the MNIST dataset, with training, evaluation, and performance comparison.

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
   - Split into training (80%) and testing (20%).

4. **Model Training**
   - Trained and compared five models:
     - 🧠 Multi-layer Perceptron (MLP)  
     - 🔤 Perceptron  
     - 🌳 Decision Tree  
     - 🌲 Random Forest  
     - ⚡ XGBoost  

5. **Evaluation**
   - Measured Accuracy, Precision, Recall, F1-score.
   - Compared results across all models.
   - Found **MLP achieved the best performance**.

---

## 📊 Results

| Model              | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| 🧠 MLPClassifier   | 0.976    | 0.97      | 0.97   | 0.97     |
| 🔤 Perceptron      | 0.920    | 0.92      | 0.92   | 0.92     |
| 🌳 Decision Tree   | 0.860    | 0.86      | 0.86   | 0.86     |
| 🌲 Random Forest   | 0.960    | 0.96      | 0.96   | 0.96     |
| ⚡ XGBoost         | 0.968    | 0.97      | 0.97   | 0.97     |

> ✅ **Best Model:** MLPClassifier – achieved the highest performance across all metrics.

---

## 🚀 Conclusion

- This project shows the impact of different ML algorithms on handwritten digit recognition.  
- Simple models (Perceptron, Decision Tree) gave decent results but lacked generalization.  
- Ensemble methods (Random Forest, XGBoost) performed strongly.  
- Neural networks (MLP) achieved the best accuracy and overall performance, confirming that deep learning approaches are superior for image recognition