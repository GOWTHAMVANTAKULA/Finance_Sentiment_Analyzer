
# Project Overview

This project focuses on Financial Sentiment Analysis by comparing multiple Natural Language Processing (NLP) and Deep Learning models to classify financial text into:

1. Positive
2. Neutral
3. Negative

The main goal was to evaluate different approaches and identify the best-performing model for real-world deployment.

## 📂 Dataset

The dataset contains financial news statements labeled into three sentiment categories:

Label	Meaning
 -- 0	Neutral
-- 1	Positive
-- 2	Negative

📊 Class Distribution

Neutral: 2879
Positive: 1363
Negative: 604

⚠️ Note: The dataset is imbalanced, which impacts model performance.

## 🧠 Models Implemented

## 1️⃣ TF-IDF + Logistic Regression
* Traditional machine learning approach
* Converts text into numerical vectors using TF-IDF
### Performance:
* Accuracy: 77%
* Weak recall for minority classes (negative & positive)
  
## 2️⃣ RNN (Recurrent Neural Network)
* Sequential deep learning model
### Performance:
* Accuracy: 64%
* Poor performance due to vanishing gradient & limited context understanding

## 3️⃣ LSTM (Long Short-Term Memory)
* Improved version of RNN for handling sequences
### Performance:
* Accuracy: 75%
* Better than RNN but still limited in capturing deep context
  
## 4️⃣ Transformer (BERT) ✅ (Selected Model)
* Pre-trained transformer-based model
* Fine-tuned for sequence classification using PyTorch
### Performance:
* Accuracy: 84%
* Best performance across all models
* Strong recall for all classes, including minority classes
  
## 📊 Model Comparison
### Model	Accuracy	Key Insight
* RNN	64%	Poor context understanding
* TF-IDF + LR	77%	Good baseline but lacks semantics
* LSTM	75%	Better sequence learning
* BERT (Transformer)	84%	Best contextual understanding
  
## 🎯 Final Model Selection

The Transformer (BERT) model was selected because:

✔️ Highest accuracy

✔️ Better handling of class imbalance

✔️ Strong contextual understanding

✔️ Industry-standard approach

## 🛠️ Tech Stack
* Python
* PyTorch
* Transformers (Hugging Face)
* Scikit-learn
* Pandas, NumPy
* Streamlit (for UI)
  
## 💻 Project Structure

project/
│
├── models/
│   ├── bert_model/
│
├── notebooks/
│   ├── ml_models.ipynb
│   ├── transformer_model.ipynb
│
├── app.py          # Streamlit app
├── requirements.txt
└── README.md

## 🎯 How It Works

Input Text

   ↓
   
Tokenizer (BERT)

   ↓
   
Convert to input_ids + attention_mask

   ↓
   
BERT Model

   ↓
   
Logits

   ↓
   
Prediction (Positive / Neutral / Negative)

## 🌐 Streamlit Application

### A simple UI was built using Streamlit:

Features:
* Input financial text
* Predict sentiment instantly
* Clean and interactive interface
### Run Locally:

# pip install -r requirements.txt
# streamlit run app.py

## 📈 Key Learnings
* Transformers outperform traditional ML and RNN-based models in NLP tasks
* Contextual understanding is critical for sentiment analysis
* Handling class imbalance is important for real-world datasets
* End-to-end pipeline (training → evaluation → deployment) is essential

## 🚀 Future Improvements
* Hyperparameter tuning for better accuracy
* Use advanced models like RoBERTa
* Deploy using FastAPI for production
* Add real-time financial news integration

## 🙌 Conclusion
This project demonstrates a complete NLP workflow:
* Data preprocessing
* Model comparison
* Performance evaluation
* Deployment

👉 Final outcome: A production-ready sentiment analysis system using Transformers
