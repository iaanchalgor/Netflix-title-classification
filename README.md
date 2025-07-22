# Netflix-title-classification


This project focuses on building a machine learning model to classify whether a Netflix title is a **Movie** or a **TV Show** based on its metadata and content-related features.

## 📌 Problem Statement

Netflix hosts thousands of titles categorized into Movies and TV Shows. The goal of this project is to classify each title based on engineered features using machine learning techniques.

---

## 🧰 Technologies Used

- Python 🐍  
- Pandas & NumPy for data manipulation  
- Scikit-learn for machine learning  
- Matplotlib & Seaborn for data visualization  
- Jupyter Notebook for interactive development  
- Pickle for model serialization  

---

## 📊 Dataset

The dataset is sourced from public Netflix title data and includes the following fields:

- Title
- Type (Movie / TV Show) — **Target**
- Director
- Cast
- Country
- Date Added
- Release Year
- Rating
- Duration
- Genres
- Description

---

## 🧪 Feature Engineering

To improve model performance, multiple feature transformations were performed:

1. **Duration-based**: Numeric transformation (in minutes or seasons)
2. **Rating-based**: Categorical to numerical mapping
3. **Genre-based**: Extracting top genres
4. **Description-based**: Text length, keyword flags
5. **Director**: Frequency count encoding

---

## 🧠 Model Building

The Logistic Regression model was chosen for classification due to its simplicity and interpretability.

- **Split**: 80% Train / 20% Test
- **Accuracy**: Achieved **~99% accuracy**
- **Evaluation**: Used confusion matrix, precision, recall, and F1-score

---

## ✅ Results

- **High accuracy (~99%)** on validation set
- **Confusion Matrix** shows model handles class imbalance well
- Feature engineering significantly boosted model performance

---

## 💾 Model Export

The trained model is saved using `pickle` and can be loaded later for predictions.

```python
import pickle

with open("netflix_logistic_model.pkl", "wb") as file:
    pickle.dump(logreg, file)
