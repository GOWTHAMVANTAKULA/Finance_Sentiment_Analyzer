import streamlit as st
import torch
import json
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="📊 FinSentiment AI", layout="centered")

# -----------------------------
# Load Model 
# -----------------------------
def load_model():
    tokenizer = BertTokenizer.from_pretrained("tokenizer")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Load Labels
# -----------------------------
with open("label_map.json") as f:
    label_map = json.load(f)

id2label = {v: k for k, v in label_map.items()}

# -----------------------------
# UI Header
# -----------------------------
st.markdown("<h1 style='text-align:center;'>📊 FinSentiment AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze Financial News Sentiment with AI</p>", unsafe_allow_html=True)

# -----------------------------
# Example Inputs
# -----------------------------
with st.expander("💡 Try Example Inputs"):
    st.write("Positive: company profits increased significantly")
    st.write("Negative: company reported huge losses and bankruptcy")
    st.write("Neutral: company announced quarterly results")

# -----------------------------
# Input Box
# -----------------------------
text = st.text_area("✍️ Enter Financial News Text", height=120)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    
    else:
        with st.spinner("Analyzing sentiment... ⏳"):

            # SAME preprocessing (important)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

        label = id2label[pred]

        # -----------------------------
        # Styling
        # -----------------------------
        color_map = {
            "positive": "#28a745",
            "neutral": "#007bff",
            "negative": "#dc3545"
        }

        emoji_map = {
            "positive": "🚀",
            "neutral": "😐",
            "negative": "📉"
        }

        # -----------------------------
        # Prediction Card
        # -----------------------------
        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:10px;
                text-align:center;
                background-color:#f5f5f5;
                margin-top:20px;
            ">
                <h2 style="color:{color_map[label]};">
                    {emoji_map[label]} {label.upper()}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Confidence Scores
        # -----------------------------
        st.subheader("📊 Confidence Scores")

        for i, prob in enumerate(probs[0]):
            st.progress(float(prob))
            st.write(f"{id2label[i]}: {prob.item():.2f}")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write("Model: BERT (bert-base-uncased)")
st.sidebar.write("Task: Financial Sentiment Analysis")
st.sidebar.write("Classes: Positive 🚀 | Neutral 😐 | Negative 📉")

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit")




