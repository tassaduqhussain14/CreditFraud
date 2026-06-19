"""
streamlit_app.py  —  Credit Fraud Detection Dashboard
------------------------------------------------------
Usage:
  streamlit run streamlit_app.py
"""
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

#Config

API_URL = "http://localhost:8000"
V_COLS = [f"V{i}" for i in range(1, 29)]
ALL_COLS = ["Time"] + V_COLS + ["Amount"]

#Page Setup

st.set_page_config(
    page_title="💳 Credit Fraud Detector",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection")
st.caption("XGBoost model FastAPI backend")

# ─── Sidebar ─────

st.sidebar.header("⚙️ Settings")
api_base = st.sidebar.text_input("FastAPI URL", value=API_URL)

# Health check
try:
    resp = requests.get(f"{api_base}/", timeout=3)
    info = resp.json()
    if info.get("model_loaded"):
        st.sidebar.success("✅ API Connected Model Ready")
    else:
        st.sidebar.warning("⚠️ API Connected Model NOT loaded")
except Exception:
    st.sidebar.error("❌ Connection Failed")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Single Check", "📦 Batch Upload", "📖 API Docs"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Transaction
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Single Transaction Check")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Basic Info**")
        time_val = st.number_input("Time (seconds)", value=406.0, step=1.0)
        amount_val = st.number_input("Amount ($)", value=149.62, min_value=0.0, step=0.01)

        st.markdown("**V1 – V14** (PCA Features)")
        v_vals = {}
        cols_a = st.columns(2)
        for i, vi in enumerate(V_COLS[:14]):
            with cols_a[i % 2]:
                v_vals[vi] = st.number_input(vi, value=round(float(np.random.randn()), 4),
                                              format="%.4f", key=f"s_{vi}")

    with col_right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("**V15 – V28** (PCA Features)")
        cols_b = st.columns(2)
        for i, vi in enumerate(V_COLS[14:]):
            with cols_b[i % 2]:
                v_vals[vi] = st.number_input(vi, value=round(float(np.random.randn()), 4),
                                              format="%.4f", key=f"s2_{vi}")

    if st.button("🔍 Check Transaction", type="primary", use_container_width=True):
        payload = {"Time": time_val, "Amount": amount_val, **v_vals}
        try:
            r = requests.post(f"{api_base}/predict", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()

            prob = res["fraud_probability"]
            label = res["label"]
            conf = res["confidence"]

            st.markdown("---")
            st.subheader("🔎 Result")

            if label == "Fraud":
                st.error(f"🚨 **FRAUD DETECTED** Probability: {prob*100:.1f}%")
            else:
                st.success(f"✅ **LEGITIMATE** Fraud Probability: {prob*100:.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", label)
            c2.metric("Fraud Probability", f"{prob*100:.2f}%")
            c3.metric("Confidence", conf)

            st.progress(prob)

        except requests.RequestException as e:
            st.error(f"API Error: {e}")


# TAB 2 — Batch Upload

with tab2:
    st.subheader("Batch CSV Upload")
    st.info(
        "CSV file upload with same columns: "
        "`Time, V1–V28, Amount` (Class column optional)."
    )

    uploaded = st.file_uploader("Choose CSV File", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Rows loaded:** {len(df_up)}")
        st.dataframe(df_up.head(5), use_container_width=True)

        # Check required columns
        missing = [c for c in ALL_COLS if c not in df_up.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Predictions is running..."):
                    batch_rows = df_up[ALL_COLS].to_dict(orient="records")
                    payload = {"transactions": batch_rows}
                    try:
                        r = requests.post(
                            f"{api_base}/predict-batch",
                            json=payload, timeout=60
                        )
                        r.raise_for_status()
                        batch_res = r.json()

                        results_df = pd.DataFrame(batch_res["results"])
                        df_up["Prediction"] = results_df["label"].values
                        df_up["Fraud_Probability"] = results_df["fraud_probability"].values
                        df_up["Confidence"] = results_df["confidence"].values

                        # Summary
                        total = batch_res["total"]
                        fraud_n = batch_res["fraud_count"]
                        legit_n = batch_res["legit_count"]

                        st.markdown("### 📊 Batch Results")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total", total)
                        m2.metric("Fraud", fraud_n, delta=f"{fraud_n/total*100:.1f}%")
                        m3.metric("Legit", legit_n)

                        st.dataframe(
                            df_up[["Time", "Amount", "Prediction", "Fraud_Probability", "Confidence"]],
                            use_container_width=True
                        )

                        csv_out = df_up.to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Results",
                            csv_out,
                            "fraud_predictions.csv",
                            "text/csv",
                        )

                    except requests.RequestException as e:
                        st.error(f"API Error: {e}")

# ═════════════════════════
# TAB 3 — API Docs Link
# ═════════════════════════

with tab3:
    st.subheader("📖 Interactive API Docs")
    st.markdown(
        f"FastAPI generate automatically API docs:\n\n"
        f"- **Swagger UI:** [{api_base}/docs]({api_base}/docs)\n"
        f"- **ReDoc:**      [{api_base}/redoc]({api_base}/redoc)\n\n"
    )
    st.code(
        """# Command line se test karo:
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"Time": 406, "Amount": 149.62, "V1": -1.36, ...}'
""",
        language="bash",
    )
