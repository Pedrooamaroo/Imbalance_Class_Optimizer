# ⚖️ Class Imbalance Optimizer: Custom Cost-Sensitive Logistic Regression Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Numpy](https://img.shields.io/badge/Numpy-From_Scratch-orange.svg)](https://numpy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-green.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview
**Class Imbalance Optimizer** is a comprehensive machine learning study and pipeline designed to solve binary classification problems in highly imbalanced datasets. 

Unlike standard implementations using scikit-learn, this project builds a **custom Logistic Regression algorithm from scratch**, incorporating a cost-sensitive loss function specifically designed to penalize errors on minority classes. The pipeline automatically ingests, cleans, and evaluates performance across **50 distinct benchmark datasets**, comparing algorithmic-level solutions (Class Weighting) against data-level solutions (SMOTE).

---

## 📈 Executive Performance Summary
The study evaluated the models across metrics critical for imbalance problems (Recall, F1-Score, and ROC AUC). The custom cost-sensitive approach significantly outperformed the baseline, while combining it with SMOTE yielded the highest ability to distinguish classes.

| Model Architecture | Precision | Recall | F1-Score | ROC AUC | Improvement (AUC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom LR + SMOTE** | 0.359 | **0.845** | **0.477** | **0.881** | **🏆 Best (+52%)** |
| Custom LR (Weighted) | **0.707** | 0.429 | 0.490 | 0.709 | +22% |
| Standard LR (Baseline) | 0.167 | 0.507 | 0.226 | 0.579 | Baseline |

**Key Takeaway:** While the weighted loss function improved precision, the introduction of synthetic data (SMOTE) was necessary to maximize Recall and AUC, effectively solving the "minority class ignorance" problem inherent in standard algorithms.

---

## 🔬 Technical Deep Dive

### 1. The "Smart" Preprocessing Engine
To handle 50 diverse datasets without manual intervention, I built an automated cleaning class:
* **Dynamic Type Detection:** Automatically separates features into Binary, Categorical, and Numerical.
* **Smart Imputation:** * *Categorical/Binary:* Imputed using Mode.
    * *Numerical:* Imputed using K-Nearest Neighbors (KNN).
* **Encoding:** Auto-switches between Label Encoding (for high cardinality) and One-Hot Encoding based on unique value thresholds.

### 2. Algorithmic Implementation (From Scratch)
Instead of using `sklearn.linear_model`, I implemented the math manually to allow for custom loss manipulation:
* **Gradient Descent:** Implemented using `autograd` to compute derivatives of the loss function.
* **Custom Loss Function:** Modified Binary Cross Entropy to include an `imbalance_penalty`:
  
  $$J(\theta) = - \frac{1}{N} \sum [ w_{pos} \cdot y \log(\hat{y}) + w_{neg} \cdot (1-y) \log(1-\hat{y}) ]$$
  
  *Where $w$ is dynamically calculated based on the imbalance ratio of the specific dataset being processed.*

---

## 🛠️ Installation & Usage

### 1. Setup
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/pedrooamaroo/Imbalance_Class_Optimizer.git](https://github.com/pedrooamaroo/Imbalance_Class_Optimizer.git)
cd Imbalance_Class_Optimizer
pip install -r requirements.txt
