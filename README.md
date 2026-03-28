# Fatigue Prediction using Machine Learning

## Overview

Fatigue is a common but subjective symptom that is difficult to quantify. This project uses machine learning to predict fatigue based on user-reported symptoms.

Due to the sparse and imbalanced nature of the dataset, the problem was formulated as a binary classification task.

---

## Dataset

* **Source:** Flaredown Autoimmune Symptom Tracker
* Contains user-reported symptoms with severity levels
* Highly imbalanced (most entries show no fatigue)

---

## Approach

* Preprocessed and pivoted symptom data into structured features
* Removed low-frequency symptoms to reduce noise
* Converted fatigue into binary classification (0 = no fatigue, 1 = fatigue)
* Applied **oversampling** to handle class imbalance
* Trained a **Random Forest Classifier**
* Performed **threshold tuning** to balance recall and precision

---

## Results

* **Accuracy:** ~36%
* **Recall (Fatigue):** 0.94
* **Precision (Fatigue):** 0.04

The model prioritizes detecting fatigue cases (high recall), ensuring minimal missed instances, at the cost of increased false positives.

---

## Example Prediction

```python
current_symptoms = {
    'Headache': 3,
    'Nausea': 2,
    'Dizziness': 2,
    'Anxiety': 3
}

# Output:
Prediction: 1 (Fatigue detected)
Confidence: 0.59
```

---

## Key Insights

* Fatigue prediction is challenging due to weak correlation between symptoms
* Significant overlap exists between fatigue and non-fatigue cases
* Model performance is heavily influenced by threshold selection

---

## Limitations

* Sparse and noisy dataset
* High false positives
* Limited feature representation

---

## Future Improvements

* Feature engineering
* Temporal analysis of symptom progression
* Advanced models (boosting, deep learning)

---

## How to Run

1. Open the notebook in Jupyter or Kaggle
2. Run all cells sequentially
3. Modify the input symptoms in the prediction section

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
