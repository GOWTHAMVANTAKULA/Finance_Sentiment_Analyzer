# 📊 FinSentiment AI : NLP-Based Sentiment Analysis : Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red?logo=streamlit)
![BERT](https://img.shields.io/badge/Model-BERT-orange)

## 📌 Project Overview

An end-to-end NLP project that classifies financial news statements into **Positive**, **Neutral**, or **Negative** sentiment. The project compares 4 different NLP approaches — from traditional ML to state-of-the-art Transformer models — and deploys the best model as an interactive web application.

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Total Samples | 4,846 |
| Train Split | 3,876 (80%) |
| Test Split | 970 (20%) |
| Classes | Positive, Neutral, Negative |

**Class Distribution:**

| Label | Count |
|-------|-------|
| Neutral | 2,879 |
| Positive | 1,363 |
| Negative | 604 |

> ⚠️ Dataset is imbalanced — Neutral class dominates. BERT handled this best.

**Sample Data:**
```
neutral  → "According to Gran, the company has no plans t..."
negative → "The international electronic industry company..."
positive → "With the new production plant the company woul..."
```

---

## 🧠 Models Implemented & Compared

### 1️⃣ TF-IDF + Logistic Regression (Baseline)

```
              precision    recall  f1-score   support

   negative       0.90      0.48      0.63       110
    neutral       0.75      0.95      0.84       571
   positive       0.80      0.52      0.63       289

    accuracy                           0.77       970
   macro avg       0.82      0.65      0.70       970
weighted avg       0.78      0.77      0.75       970
```

### 2️⃣ RNN (Recurrent Neural Network)

```
              precision    recall  f1-score   support

   negative       0.47      0.40      0.43       110
    neutral       0.73      0.83      0.78       571
   positive       0.46      0.37      0.41       289

    accuracy                           0.64       970
   macro avg       0.56      0.53      0.54       970
weighted avg       0.62      0.64      0.63       970
```

### 3️⃣ LSTM (Long Short-Term Memory)

```
              precision    recall  f1-score   support

   negative       0.71      0.61      0.65       110
    neutral       0.77      0.88      0.82       571
   positive       0.69      0.54      0.61       289

    accuracy                           0.75       970
   macro avg       0.72      0.67      0.69       970
weighted avg       0.74      0.75      0.74       970
```

### 4️⃣ BERT Transformer ✅ (Selected Model)

```
              precision    recall  f1-score   support

   negative       0.90      0.81      0.85       110
    neutral       0.90      0.85      0.87       571
   positive       0.74      0.84      0.79       289

    accuracy                           0.84       970
   macro avg       0.85      0.83      0.84       970
weighted avg       0.85      0.84      0.85       970
```

---

## 📊 Model Comparison Summary

| Model | Accuracy | Macro F1 | Key Insight |
|-------|----------|----------|-------------|
| RNN | 64% | 0.54 | Poor context understanding |
| LSTM | 75% | 0.69 | Better but limited context |
| TF-IDF + LR | 77% | 0.70 | Good baseline, lacks semantics |
| **BERT ✅** | **84%** | **0.84** | Best contextual understanding |

---

## 🏆 Why BERT Won

- ✅ Highest accuracy across all classes
- ✅ Best handling of class imbalance
- ✅ Strong recall for minority class (negative: 81%)
- ✅ Contextual word embeddings capture financial nuance
- ✅ Pre-trained on massive corpus — fine-tuned efficiently

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.1.0 |
| NLP | HuggingFace Transformers 4.35.0 |
| Model | BERT (bert-base-uncased) |
| ML Baseline | Scikit-learn (TF-IDF + LR) |
| Frontend | Streamlit 1.35.0 |
| Data | Pandas, NumPy |

---

## 📁 Project Structure

```
Finance_Sentiment_Analyzer/
├── model/
│   └── finance_sentiment_model.pth         ← fine-tuned BERT weights
├── app.py                                  ← Streamlit web application
├── requirements.txt                        ← Python dependencies
├── label_map.json                          ← label encoding map
├── Finance_Sentiment_Analyzer_BERT.ipynb   ← BERT training notebook
├── Finance_Sentiment_Analyzer_TFIDF.ipynb  ← baseline models notebook
└── README.md
```

---

## ⚙️ How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/GOWTHAMVANTAKULA/Finance_Sentiment_Analyzer.git
cd Finance_Sentiment_Analyzer
```

**2. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the app:**
```bash
streamlit run app.py --server.port 8502
```

**5. Open in browser:**
```
http://localhost:8502
```

---

## 🔍 How It Works

```
User Input (Financial Text)
         ↓
BERT Tokenizer (bert-base-uncased)
         ↓
input_ids + attention_mask
         ↓
Fine-tuned BERT Model
         ↓
Logits → Softmax
         ↓
Sentiment + Confidence Score
(Positive 🚀 / Neutral 😐 / Negative 📉)
```

---

## 📈 Key Learnings

- Transformer models significantly outperform traditional ML and RNN-based models
- Class imbalance affects recall for minority classes — BERT handles this better
- Fine-tuning pre-trained models is more effective than training from scratch
- Contextual embeddings capture financial terminology better than TF-IDF

---

## 👤 Author

**Vantakula Gowtham Naidu**

[![GitHub](https://img.shields.io/badge/GitHub-GOWTHAMVANTAKULA-black?logo=github)](https://github.com/GOWTHAMVANTAKULA)

