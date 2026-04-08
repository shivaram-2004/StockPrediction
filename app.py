
import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import plot_tree
import numpy as np

# ── MUST be the very first Streamlit call ──────────────────────────────────
st.set_page_config(page_title="Stock Decision Predictor", page_icon="📈", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stButton > button {
        background: linear-gradient(135deg, #1a6b3c, #22a35b);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #22a35b, #2ecc71);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(34,163,91,0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0d1f12, #132b1a);
        border: 2px solid #22a35b;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .section-header {
        border-left: 4px solid #22a35b;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)

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

FEATURE_NAMES = ["Day", "Trend", "RSI", "Volume", "News"]
CLASS_NAMES   = ["BUY", "SELL", "HOLD"]
CLASS_COLORS  = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#f39c12"}

# ── Title ──────────────────────────────────────────────────────────────────
st.title("📈 Stock Decision Predictor")
st.markdown(
    f"Using the **best model** trained on **{meta['best_company']}** data "
    f"with accuracy **{meta['best_accuracy']:.2f}%**"
)

# ── Layout: sidebar + main ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 All Model Accuracies")
    acc_df = pd.DataFrame(
        meta["all_accuracies"].items(),
        columns=["Company", "Accuracy (%)"]
    ).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    acc_df["Accuracy (%)"] = acc_df["Accuracy (%)"].round(2)
    
    # Highlight best
    def highlight_best(row):
        if row["Company"] == meta["best_company"]:
            return ["background-color: #1a3d2b; color: #2ecc71; font-weight: bold"] * 2
        return [""] * 2

    st.dataframe(
        acc_df.style.apply(highlight_best, axis=1),
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("### 🏆 Best Model Info")
    st.markdown(f"**Company:** {meta['best_company']}")
    st.markdown(f"**Accuracy:** {meta['best_accuracy']:.2f}%")
    st.markdown(f"**Max Depth:** {model.get_depth()}")
    st.markdown(f"**Num Leaves:** {model.get_n_leaves()}")
    st.markdown(f"**Features Used:** {model.n_features_in_}")

# ── Main content ───────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="section-header"><h3>🎛️ Enter Stock Features</h3></div>', unsafe_allow_html=True)

    day = st.slider("Day", min_value=1, max_value=50, value=1, step=1)

    trend = st.selectbox(
        "Trend",
        options=[0, 1, 2],
        format_func=lambda x: {0: "↑ Up", 1: "↓ Down", 2: "→ Stable"}[x]
    )

    rsi = st.selectbox(
        "RSI",
        options=[0, 1, 2],
        format_func=lambda x: {0: "🔵 Low", 1: "🟡 Medium", 2: "🔴 High"}[x]
    )

    volume = st.selectbox(
        "Volume",
        options=[0, 1, 2],
        format_func=lambda x: {0: "🔵 Low", 1: "🟡 Medium", 2: "🔴 High"}[x]
    )

    news = st.selectbox(
        "News Sentiment",
        options=[0, 1],
        format_func=lambda x: {0: "📰 Negative", 1: "📰 Positive"}[x]
    )

    predict_btn = st.button("🔮 Predict Decision", use_container_width=True)

with right_col:
    if predict_btn:
        input_df = pd.DataFrame(
            [[day, trend, rsi, volume, news]],
            columns=FEATURE_NAMES
        )

        prediction     = model.predict(input_df)[0]
        proba          = model.predict_proba(input_df)[0]
        label_map      = {0: "BUY 🟢", 1: "SELL 🔴", 2: "HOLD 🟡"}
        result         = label_map.get(prediction, f"Unknown ({prediction})")
        result_name    = CLASS_NAMES[prediction] if prediction < len(CLASS_NAMES) else str(prediction)
        result_color   = CLASS_COLORS.get(result_name, "#aaaaaa")

        # ── Prediction card ────────────────────────────────────────────────
        st.markdown(f"""
        <div class="prediction-box">
            <p style="color:#aaa; font-size:14px; margin-bottom:8px;">PREDICTION RESULT</p>
            <h1 style="color:{result_color}; font-size:3rem; margin:0;">{result}</h1>
            <p style="color:#888; font-size:13px; margin-top:8px;">
                Model confidence breakdown
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ───────────────────────────────────────────────
        proba_df = pd.DataFrame({
            "Decision": CLASS_NAMES[:len(proba)],
            "Probability (%)": (proba * 100).round(1)
        })
        st.dataframe(proba_df, use_container_width=True, hide_index=True)

        # ── Input summary ──────────────────────────────────────────────────
        with st.expander("📋 Input Summary"):
            st.table(pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Value": [
                    day,
                    {0: "Up", 1: "Down", 2: "Stable"}[trend],
                    {0: "Low", 1: "Medium", 2: "High"}[rsi],
                    {0: "Low", 1: "Medium", 2: "High"}[volume],
                    {0: "Negative", 1: "Positive"}[news],
                ]
            }))

# ── Decision Tree Section (full width below) ───────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header"><h3>🌳 Decision Tree Visualization</h3></div>', unsafe_allow_html=True)

show_tree = st.checkbox("Show Decision Tree", value=predict_btn)

if show_tree:
    depth_limit = st.slider(
        "Max depth to display (full tree can be large)",
        min_value=1, max_value=min(model.get_depth(), 6), value=min(3, model.get_depth())
    )

    # ── Use a LIGHT background for the tree so all node colors are readable ──
    fig, ax = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    plot_tree(
        model,
        max_depth=depth_limit,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        ax=ax,
        fontsize=10,
        precision=2,
    )

    # Do NOT override text colors — let matplotlib pick contrasting text per node
    # Only style the title
    ax.set_title(
        f"Decision Tree — {meta['best_company']}  |  depth shown: {depth_limit} / {model.get_depth()}",
        color="#111111", fontsize=13, fontweight="bold", pad=14
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="BUY"),
        mpatches.Patch(color="#e74c3c", label="SELL"),
        mpatches.Patch(color="#f39c12", label="HOLD"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              facecolor="white", edgecolor="#cccccc", labelcolor="black")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    # ── Highlight prediction path (if prediction was made) ─────────────────
    if predict_btn:
        st.markdown("---")
        st.markdown("#### 🔍 Decision Path for Your Input")

        node_indicator = model.decision_path(input_df)
        node_ids       = node_indicator.indices[
                            node_indicator.indptr[0]:node_indicator.indptr[1]
                         ]
        
        tree_     = model.tree_
        path_rows = []

        for node_id in node_ids:
            if tree_.children_left[node_id] == -1:
                path_rows.append({
                    "Node": f"#{node_id}",
                    "Type": "🍃 Leaf",
                    "Feature": "—",
                    "Condition": "—",
                    "Samples": int(tree_.n_node_samples[node_id]),
                    "Class": CLASS_NAMES[int(np.argmax(tree_.value[node_id]))]
                })
            else:
                feat      = FEATURE_NAMES[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]
                val       = input_df.iloc[0][feat]
                direction = f"{val:.0f} ≤ {threshold:.2f} → LEFT" if val <= threshold else f"{val:.0f} > {threshold:.2f} → RIGHT"
                path_rows.append({
                    "Node": f"#{node_id}",
                    "Type": "🔀 Split",
                    "Feature": feat,
                    "Condition": direction,
                    "Samples": int(tree_.n_node_samples[node_id]),
                    "Class": CLASS_NAMES[int(np.argmax(tree_.value[node_id]))]
                })

        path_df = pd.DataFrame(path_rows)
        st.dataframe(path_df, use_container_width=True, hide_index=True)
        st.caption(f"Total nodes visited: {len(node_ids)} | Final decision: **{result}**")

st.markdown("---")
st.caption("Decision classes: BUY 🟢 · SELL 🔴 · HOLD 🟡  |  Best model: " + meta.get("best_company", "N/A"))
