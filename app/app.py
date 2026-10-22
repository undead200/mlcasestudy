import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Machine Learning Capstone Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
    }
    .safe-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .danger-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find model directory
def get_model_path(filename):
    search_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "models"),
        os.path.join(os.path.dirname(__file__), "models"),
        "models",
        "../models",
        "."
    ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return filename

# Cache loaded artifacts
@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    try:
        artifacts['reg_model'] = joblib.load(get_model_path('best_regression_model.pkl'))
        artifacts['reg_prep'] = joblib.load(get_model_path('regression_preprocessor.pkl'))
        artifacts['reg_meta'] = joblib.load(get_model_path('regression_meta.pkl'))
    except Exception as e:
        artifacts['reg_error'] = str(e)

    try:
        artifacts['clf_model'] = joblib.load(get_model_path('best_classification_model.pkl'))
        artifacts['clf_scaler'] = joblib.load(get_model_path('classification_scaler.pkl'))
        artifacts['clf_meta'] = joblib.load(get_model_path('classification_meta.pkl'))
    except Exception as e:
        artifacts['clf_error'] = str(e)

    try:
        artifacts['cl_model'] = joblib.load(get_model_path('kmeans_clustering_model.pkl'))
        artifacts['cl_scaler'] = joblib.load(get_model_path('clustering_scaler.pkl'))
        artifacts['cl_meta'] = joblib.load(get_model_path('clustering_meta.pkl'))
    except Exception as e:
        artifacts['cl_error'] = str(e)

    return artifacts

artifacts = load_all_artifacts()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/brain.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select ML Track:",
    [
        "🏛️ Overview & Model Dashboard",
        "🛵 Regression: Food Delivery Time",
        "💧 Classification: Water Potability",
        "📈 Clustering: Market Regime Discovery"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Course**: 23CSE301 Machine Learning  
**Capstone Project**: 3-Track ML Pipeline  
**Deployment**: Interactive GUI (Streamlit)
""")

# ==============================================================================
# PAGE 1: OVERVIEW & DASHBOARD
# ==============================================================================
if page == "🏛️ Overview & Model Dashboard":
    st.markdown('<div class="main-header">Machine Learning Capstone Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Multi-Track Inference Engine for Regression, Classification & Clustering</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TRACK 1: REGRESSION</div>
            <div class="metric-value">0.83+ R² Score</div>
            <p style="margin-top: 5px; color: #6B7280; font-size: 0.85rem;">Predicts Food Delivery Time (min) with Gradient Boosting.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TRACK 2: CLASSIFICATION</div>
            <div class="metric-value">0.69+ ROC-AUC</div>
            <p style="margin-top: 5px; color: #6B7280; font-size: 0.85rem;">Assesses Water Potability using Tuned Random Forest.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TRACK 3: CLUSTERING</div>
            <div class="metric-value">0.44 Silhouette</div>
            <p style="margin-top: 5px; color: #6B7280; font-size: 0.85rem;">Discovers Stock Market Regimes via K-Means (k=4).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Executive Performance Summary Table")
    summary_data = {
        "Track": ["Regression", "Regression", "Classification", "Classification", "Clustering", "Clustering"],
        "Algorithm": ["Gradient Boosting (Tuned)", "Random Forest", "Random Forest (Tuned)", "Gradient Boosting", "K-Means (k=4)", "Agglomerative (Ward)"],
        "Primary Metric": ["R²: 0.8340", "R²: 0.8120", "ROC-AUC: 0.6891", "ROC-AUC: 0.6780", "Silhouette: 0.4390", "Silhouette: 0.3802"],
        "Secondary Metric": ["RMSE: 3.98 min", "RMSE: 4.21 min", "Accuracy: 66.0%", "Accuracy: 65.5%", "Davies-Bouldin: 0.7997", "Davies-Bouldin: 0.8188"],
        "Status": ["Production Tuned", "Baseline", "Production Tuned", "Baseline", "Optimal Partition", "Hierarchical Sample"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.markdown("---")
    st.markdown("""
    #### 🚀 How to Use this Application:
    1. **Choose a Track** from the left navigation sidebar.
    2. **Customize Input Parameters** using the interactive sliders and selectors (pre-filled with realistic sample defaults).
    3. **Click Predict / Discover** to run real-time inference through the trained serialized models.
    4. **Inspect Confidence & Diagnostic Cards** detailing feature attributions and safety thresholds.
    """)

# ==============================================================================
# PAGE 2: REGRESSION TRACK (Food Delivery Time)
# ==============================================================================
elif page == "🛵 Regression: Food Delivery Time":
    st.markdown('<div class="main-header">Food Delivery Time Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict estimated delivery duration based on driver attributes, weather, traffic, and distance</div>', unsafe_allow_html=True)

    if 'reg_error' in artifacts:
        st.error(f"Error loading Regression artifacts: {artifacts['reg_error']}")
    else:
        with st.form("regression_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👤 Driver & Order Parameters")
                ratings = st.slider("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.7, step=0.1)
                vehicle_condition = st.selectbox("Vehicle Condition Grade", options=[0, 1, 2, 3], index=2,
                                                 help="0 = Poor, 1 = Fair, 2 = Good, 3 = Excellent")
                multiple_deliveries = st.selectbox("Number of Concurrent Deliveries", options=[0, 1, 2, 3], index=1)
                distance_km = st.slider("Delivery Distance (km)", min_value=0.5, max_value=30.0, value=8.5, step=0.5)

            with col2:
                st.subheader("🌤️ Environmental & Transit Conditions")
                traffic = st.selectbox("Road Traffic Density",
                                       options=['Low ', 'Medium ', 'High ', 'Jam '], index=1)
                festival = st.selectbox("Festival Occurring?", options=['No ', 'Yes '], index=0)

            submitted = st.form_submit_button("⚡ Estimate Delivery Time", use_container_width=True)

        if submitted:
            # Construct DataFrame matching training schema with standard baseline values for removed inputs
            input_dict = {
                'Delivery_person_Age': [28],
                'Delivery_person_Ratings': [ratings],
                'Vehicle_condition': [vehicle_condition],
                'multiple_deliveries': [multiple_deliveries],
                'Distance_km': [distance_km],
                'Weatherconditions': ['Sunny'],
                'Road_traffic_density': [traffic],
                'Type_of_order': ['Meal '],
                'Type_of_vehicle': ['motorcycle '],
                'Festival': [festival],
                'City': ['Metropolitian ']
            }
            input_df = pd.DataFrame(input_dict)

            try:
                # Preprocess input features
                prep = artifacts['reg_prep']
                X_proc = prep.transform(input_df)

                # Predict
                model = artifacts['reg_model']
                pred_time = model.predict(X_proc)[0]

                st.markdown("---")
                st.subheader("⏱️ Delivery Prediction Results")

                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Estimated Delivery Time", f"{pred_time:.1f} mins")
                with res_col2:
                    st.metric("Expected Arrival Window", f"{max(5, int(pred_time-3))} - {int(pred_time+4)} mins")
                with res_col3:
                    st.metric("Transit Distance", f"{distance_km:.1f} km")

                # Contextual feedback
                if traffic.strip() == 'Jam' or festival.strip() == 'Yes':
                    st.warning("⚠️ High traffic density or festival conditions detected. Delay buffer included in model output.")
                else:
                    st.success("✓ Route conditions favorable for rapid transit delivery.")

            except Exception as e:
                st.error(f"Inference error: {e}")

# ==============================================================================
# PAGE 3: CLASSIFICATION TRACK (Water Potability)
# ==============================================================================
elif page == "💧 Classification: Water Potability":
    st.markdown('<div class="main-header">Water Quality & Potability Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate chemical and physical metrics to predict whether water is safe for human consumption</div>', unsafe_allow_html=True)

    if 'clf_error' in artifacts:
        st.error(f"Error loading Classification artifacts: {artifacts['clf_error']}")
    else:
        with st.form("classification_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🧪 Chemical Balances")
                ph = st.number_input("pH Level (WHO Safe: 6.5 - 8.5)", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
                chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=7.1, step=0.1)
                sulfate = st.number_input("Sulfate (mg/L)", min_value=50.0, max_value=500.0, value=333.0, step=1.0)

            with col2:
                st.subheader("🔬 Mineral Content")
                hardness = st.number_input("Hardness (mg/L)", min_value=50.0, max_value=350.0, value=196.0, step=1.0)
                solids = st.number_input("Total Dissolved Solids (ppm)", min_value=300.0, max_value=60000.0, value=22000.0, step=100.0)
                conductivity = st.number_input("Conductivity (μS/cm)", min_value=100.0, max_value=800.0, value=426.0, step=1.0)

            with col3:
                st.subheader("🦠 Organic & Clarity Indicators")
                organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=2.0, max_value=30.0, value=14.2, step=0.1)
                trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=130.0, value=66.3, step=0.1)
                turbidity = st.number_input("Turbidity (NTU)", min_value=1.0, max_value=7.0, value=3.9, step=0.1)

            submitted = st.form_submit_button("💧 Assess Water Potability", use_container_width=True)

        if submitted:
            # Engineered feature
            is_ph_optimal = 1 if (6.5 <= ph <= 8.5) else 0

            # Input array matching training feature order
            raw_features = np.array([[
                ph, hardness, solids, chloramines, sulfate,
                conductivity, organic_carbon, trihalomethanes, turbidity, is_ph_optimal
            ]])

            try:
                scaler = artifacts['clf_scaler']
                scaled_features = scaler.transform(raw_features)

                model = artifacts['clf_model']
                prediction = model.predict(scaled_features)[0]
                proba = model.predict_proba(scaled_features)[0]
                potable_prob = proba[1] * 100

                st.markdown("---")
                st.subheader("📋 Water Quality Verdict")

                col_res1, col_res2 = st.columns([1.5, 2])
                with col_res1:
                    if prediction == 1:
                        st.markdown('<div class="safe-badge">✓ POTABLE (SAFE TO DRINK)</div>', unsafe_allow_html=True)
                        st.write(f"**Confidence Assessment**: The model evaluates this sample as **safe** with a **{potable_prob:.1f}% probability**.")
                    else:
                        st.markdown('<div class="danger-badge">⚠️ NOT POTABLE (CONTAMINATED)</div>', unsafe_allow_html=True)
                        st.write(f"**Hazard Warning**: The sample exceeds safety thresholds. Safe potability probability is only **{potable_prob:.1f}%**.")

                with col_res2:
                    st.write("**Potability Probability Meter:**")
                    st.progress(potable_prob / 100)
                    st.write(f"*Class 0 (Unsafe): {proba[0]*100:.1f}% | Class 1 (Safe): {proba[1]*100:.1f}%*")

                st.markdown("#### 🔍 WHO Guideline Compliance Checklist:")
                chk1 = "✅" if (6.5 <= ph <= 8.5) else "❌"
                chk2 = "✅" if (turbidity < 5.0) else "❌"
                chk3 = "✅" if (sulfate < 400.0) else "❌"
                st.write(f"- {chk1} **pH Level**: {ph:.1f} (Guideline: 6.5 – 8.5)")
                st.write(f"- {chk2} **Turbidity**: {turbidity:.1f} NTU (Guideline: < 5.0 NTU)")
                st.write(f"- {chk3} **Sulfate**: {sulfate:.1f} mg/L (Guideline: < 400 mg/L)")

            except Exception as e:
                st.error(f"Inference error: {e}")

# ==============================================================================
# PAGE 4: CLUSTERING TRACK (Market Regimes)
# ==============================================================================
elif page == "📈 Clustering: Market Regime Discovery":
    st.markdown('<div class="main-header">Stock Market Regime Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised partition of stock trading characteristics into distinct valuation and liquidity regimes</div>', unsafe_allow_html=True)

    if 'cl_error' in artifacts:
        st.error(f"Error loading Clustering artifacts: {artifacts['cl_error']}")
    else:
        with st.form("clustering_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💵 Intraday Price Parameters ($)")
                open_price = st.number_input("Open Price ($)", min_value=1.0, max_value=500.0, value=125.5, step=0.5)
                high_price = st.number_input("High Price ($)", min_value=1.0, max_value=500.0, value=128.0, step=0.5)
                low_price = st.number_input("Low Price ($)", min_value=1.0, max_value=500.0, value=124.0, step=0.5)
                close_price = st.number_input("Close Price ($)", min_value=1.0, max_value=500.0, value=127.2, step=0.5)

            with col2:
                st.subheader("📊 Liquidity & Volume Parameters")
                volume = st.number_input("Trading Volume (shares)", min_value=1000, max_value=1000000000, value=25000000, step=100000)
                st.info("ℹ️ Volume is automatically log-compressed via log(1 + Volume) to mirror the training preprocessing pipeline.")

            submitted = st.form_submit_button("🔍 Identify Market Regime", use_container_width=True)

        if submitted:
            # Transform volume
            log_vol = np.log1p(volume)
            raw_sample = np.array([[open_price, high_price, low_price, close_price, log_vol]])

            try:
                scaler = artifacts['cl_scaler']
                scaled_sample = scaler.transform(raw_sample)

                model = artifacts['cl_model']
                cluster_id = model.predict(scaled_sample)[0]

                # Regime interpretation profiles
                regime_descriptions = {
                    0: ("High-Volume Growth / Mid-Tier Valuation", "Characterized by moderate stock prices accompanied by heavy institutional volume turnover. Common during momentum transitions."),
                    1: ("Mega-Cap Premium Valuation Regime", "Characterized by premium stock valuations with deep, established liquidity profiles (e.g. MSFT / AAPL mature bull cycles)."),
                    2: ("Low-Price / High-Volatility Speculative Regime", "Characterized by lower absolute share prices and sporadic trading spikes typical of early-stage growth assets."),
                    3: ("Defensive / Low-Volume Consolidation Regime", "Characterized by steady trading with below-average volatility and subdued institutional turnover.")
                }

                title, desc = regime_descriptions.get(cluster_id, ("Standard Market Regime", "Standard market behavior profile."))

                st.markdown("---")
                st.subheader("🎯 Discovered Market Regime Assignment")

                res_c1, res_c2 = st.columns([1, 2])
                with res_c1:
                    st.metric("Assigned Cluster", f"Cluster {cluster_id}")
                with res_c2:
                    st.markdown(f"**Profile**: {title}")
                    st.write(desc)

                st.markdown("#### 📊 Model Performance Baseline:")
                metrics = artifacts['cl_meta'].get('metrics', {})
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Silhouette Score", f"{metrics.get('silhouette', 0.439):.4f}")
                mc2.metric("Davies-Bouldin Index", f"{metrics.get('davies_bouldin', 0.800):.4f}")
                mc3.metric("Calinski-Harabasz", f"{metrics.get('calinski_harabasz', 13717):.1f}")

            except Exception as e:
                st.error(f"Inference error: {e}")
