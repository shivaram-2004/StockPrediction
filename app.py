"""
app.py  –  Stock Decision Predictor (Streamlit)
------------------------------------------------
Run:  streamlit run app.py
"""

import os
import pickle
import pandas as pd
import streamlit as st

# ── MUST be the very first Streamlit call ──────────────────────────────────
st.set_page_config(page_title="Stock Decision Predictor", page_icon="📈", layout="centered")

# ── Load model & metadata ──────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml-pbl")

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, meta

model, meta = load_model()

# ── Title ──────────────────────────────────────────────────────────────────
st.title("📈 Stock Decision Predictor")
st.markdown(
    f"Using the **best model** trained on **{meta['best_company']}** data "
    f"with accuracy **{meta['best_accuracy']:.2f}%**"
)

# ── Sidebar – accuracy table ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 All Model Accuracies")
    acc_df = pd.DataFrame(
        meta["all_accuracies"].items(),
        columns=["Company", "Accuracy (%)"]
    ).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    acc_df["Accuracy (%)"] = acc_df["Accuracy (%)"].round(2)
    st.dataframe(acc_df, use_container_width=True)

# ── Input form ─────────────────────────────────────────────────────────────
st.subheader("Enter Stock Features")

col1, col2 = st.columns(2)

with col1:
    day = st.slider("Day", min_value=1, max_value=50, value=1, step=1)

    trend = st.selectbox(
        "Trend",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 – Up", 1: "1 – Down", 2: "2 – Stable"}[x]
    )

    rsi = st.selectbox(
        "RSI",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 – Low", 1: "1 – Medium", 2: "2 – High"}[x]
    )

with col2:
    volume = st.selectbox(
        "Volume",
        options=[0, 1, 2],
        format_func=lambda x: {0: "0 – Low", 1: "1 – Medium", 2: "2 – High"}[x]
    )

    news = st.selectbox(
        "News Sentiment",
        options=[0, 1],
        format_func=lambda x: {0: "0 – Negative", 1: "1 – Positive"}[x]
    )

# ── Predict ────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Decision", use_container_width=True):
    input_df = pd.DataFrame(
        [[day, trend, rsi, volume, news]],
        columns=["Day", "Trend", "RSI", "Volume", "News"]
    )

    prediction = model.predict(input_df)[0]

    label_map = {0: "BUY 🟢", 1: "SELL 🔴", 2: "HOLD 🟡"}
    result = label_map.get(prediction, f"Unknown ({prediction})")

    st.markdown("---")
    st.subheader("📋 Prediction Result")
    st.markdown(f"## {result}")

    with st.expander("See input summary"):
        st.table(pd.DataFrame({
            "Feature": ["Day", "Trend", "RSI", "Volume", "News"],
            "Value": [
                day,
                {0: "Up", 1: "Down", 2: "Stable"}[trend],
                {0: "Low", 1: "Medium", 2: "High"}[rsi],
                {0: "Low", 1: "Medium", 2: "High"}[volume],
                {0: "Negative", 1: "Positive"}[news],
            ]
        }))

st.markdown("---")
st.caption("Decision classes: 0 = BUY · 1 = SELL · 2 = HOLD")