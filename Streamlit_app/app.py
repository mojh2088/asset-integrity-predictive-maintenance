# ---------------------------------------------------------
# Streamlit App - RUL Prediction Frontend (Final Version)
# ---------------------------------------------------------

import streamlit as st
import requests
import json
import os

# ---------------------------------------------------------
# Load Feature Names (Same order used by the API)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your structure:
# project/
#    Streamlit_app/app.py   → BASE_DIR
#    models/feature_names.json → one level up
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.json")

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)

# ---------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict_rul"   # Adjust if running on another host/port

st.title("🔧 Remaining Useful Life (RUL) Prediction")
st.markdown("Enter sensor data below and click **Predict RUL**.\n\nThis app uses:")
st.markdown("- **XGBoost Model**")
st.markdown("- **Neural Network Model**")
st.markdown("- **Ensemble Model (Average)**")

st.divider()

# ---------------------------------------------------------
# Sidebar: Auto-fill demo data
# ---------------------------------------------------------
st.sidebar.header("⚙️ Demo Options")

if st.sidebar.button("Auto-fill sample input"):
    st.session_state.inputs = {f: 0.5 for f in feature_names}
else:
    if "inputs" not in st.session_state:
        st.session_state.inputs = {f: 0.0 for f in feature_names}

# ---------------------------------------------------------
# Main input form
# ---------------------------------------------------------
st.subheader("📌 Input Sensor/Operational Data")

cols = st.columns(3)
for idx, feat in enumerate(feature_names):
    with cols[idx % 3]:
        st.session_state.inputs[feat] = st.number_input(
            feat,
            value=float(st.session_state.inputs.get(feat, 0.0)),
            format="%.4f"
        )

# ---------------------------------------------------------
# Predict Button
# ---------------------------------------------------------
st.divider()
if st.button("🔮 Predict RUL", use_container_width=True):

    with st.spinner("Running prediction..."):

        try:
            response = requests.post(API_URL, json=st.session_state.inputs)

            if response.status_code == 200:
                result = response.json()

                xgb_rul = result["xgb_rul"]
                nn_rul = result["nn_rul"]
                ensemble_rul = result["ensemble_rul"]

                st.success("Prediction completed!")

                st.metric("XGBoost RUL", f"{xgb_rul} cycles")
                st.metric("Neural Network RUL", f"{nn_rul} cycles")
                st.metric("Ensemble RUL", f"{ensemble_rul} cycles")

                st.subheader("📊 Model Comparison")
                st.bar_chart({
                    "XGBoost": [xgb_rul],
                    "Neural Net": [nn_rul],
                    "Ensemble": [ensemble_rul]
                })

            else:
                st.error(f"API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Make sure FastAPI is running:")
            st.code("uvicorn rul_api:app --reload")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("Developed for Predictive Maintenance - CMAPSS FD001")
