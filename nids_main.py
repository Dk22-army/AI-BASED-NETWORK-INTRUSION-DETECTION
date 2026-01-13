"""
AI-Based Network Intrusion Detection System (NIDS)
Technical Implementation using Random Forest Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="AI-Based NIDS",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.0
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


def load_data(use_real_data=True, file_path='Wednesday-workingHours.pcap_ISCX.csv'):
    """
    Load network traffic data. 
    Default: Real CIC-IDS2017 dataset.
    Fallback: Simulation mode with synthetic data.
    """
    if use_real_data and file_path:
        try:
            # Load the CIC-IDS2017 dataset
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Select relevant features for training
            selected_features = [
                'Flow Duration',
                'Total Fwd Packets',
                'Total Backward Packets',
                'Flow Bytes/s',
                'Flow Packets/s',
                'Flow IAT Mean',
                'Fwd Packet Length Mean',
                'Bwd Packet Length Mean',
                'Packet Length Mean',
                'Average Packet Size'
            ]
            
            # Check if required columns exist
            available_features = [f for f in selected_features if f in df.columns]
            
            if len(available_features) < 5:
                st.warning(f"Only {len(available_features)} features found. Falling back to simulation mode.")
                raise ValueError("Insufficient features in dataset")
            
            # Keep only selected features and label
            df_filtered = df[available_features + [' Label']].copy()
            
            # Clean the data
            df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)
            df_filtered = df_filtered.dropna()
            
            # Convert label to binary: BENIGN = 0, Any attack = 1
            df_filtered['label'] = df_filtered[' Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
            df_filtered = df_filtered.drop(' Label', axis=1)
            
            # Rename columns to match simplified names
            column_mapping = {
                'Flow Duration': 'flow_duration',
                'Total Fwd Packets': 'total_fwd_packets',
                'Total Backward Packets': 'total_bwd_packets',
                'Flow Bytes/s': 'flow_bytes_per_sec',
                'Flow Packets/s': 'flow_packets_per_sec',
                'Flow IAT Mean': 'flow_iat_mean',
                'Fwd Packet Length Mean': 'fwd_packet_length_mean',
                'Bwd Packet Length Mean': 'bwd_packet_length_mean',
                'Packet Length Mean': 'packet_length_mean',
                'Average Packet Size': 'avg_packet_size'
            }
            df_filtered = df_filtered.rename(columns=column_mapping)
            
            # Sample data if too large (for performance)
            if len(df_filtered) > 50000:
                st.info(f"Dataset has {len(df_filtered)} samples. Sampling 50,000 for performance...")
                df_filtered = df_filtered.sample(n=50000, random_state=42)
            
            st.success(f"✅ Loaded real CIC-IDS2017 dataset: {len(df_filtered)} samples")
            attack_count = df_filtered['label'].sum()
            benign_count = len(df_filtered) - attack_count
            st.info(f"📊 Benign: {benign_count} | Attack: {attack_count}")
            
            return df_filtered
            
        except Exception as e:
            st.error(f"Error loading real data: {e}")
            st.info("Falling back to simulation mode...")
    
    # Simulation Mode: Generate synthetic network traffic data
    np.random.seed(42)
    n_samples = 5000
    
    # Feature generation
    data = {
        'flow_duration': np.random.randint(0, 120000, n_samples),
        'total_fwd_packets': np.random.randint(0, 100, n_samples),
        'total_bwd_packets': np.random.randint(0, 100, n_samples),
        'flow_bytes_per_sec': np.random.uniform(0, 1000000, n_samples),
        'flow_packets_per_sec': np.random.uniform(0, 1000, n_samples),
        'flow_iat_mean': np.random.uniform(0, 50000, n_samples),
        'fwd_packet_length_mean': np.random.uniform(0, 1500, n_samples),
        'bwd_packet_length_mean': np.random.uniform(0, 1500, n_samples),
        'packet_length_mean': np.random.uniform(0, 1500, n_samples),
        'avg_packet_size': np.random.uniform(0, 1500, n_samples),
    }
    
    # Create labels (0 = Normal, 1 = Attack)
    # Inject attack patterns based on feature combinations
    labels = []
    for i in range(n_samples):
        # Attack heuristics:
        # - High packet rate
        # - Unusual flow characteristics
        # - Large flow duration with few packets
        if (data['flow_packets_per_sec'][i] > 800 or 
            data['packet_length_mean'][i] > 1400 or
            (data['flow_duration'][i] > 100000 and data['total_fwd_packets'][i] < 5)):
            labels.append(1)  # Attack
        else:
            labels.append(0)  # Normal
    
    data['label'] = labels
    df = pd.DataFrame(data)
    
    st.warning("⚠️ Using simulated data (no CSV file found)")
    
    return df


def train_model(df):
    """
    Train Random Forest Classifier on network traffic data.
    """
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    with st.spinner('Training Random Forest model...'):
        rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return rf_model, accuracy, report, cm, X_test.columns


def predict_traffic(model, features, feature_names):
    """
    Predict if network traffic is normal or an attack.
    """
    # Create DataFrame with proper feature names
    input_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    return prediction, probability


# ===========================
# STREAMLIT UI
# ===========================

# Title and Header
st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Control Panel")
st.sidebar.markdown("### Model Training")

if st.sidebar.button("🚀 Train Model Now"):
    with st.spinner("Loading data and training model..."):
        # Load data (using real CSV dataset)
        data = load_data(use_real_data=True)
        
        # Train model
        model, acc, report, cm, feature_names = train_model(data)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.accuracy = acc
        st.session_state.report = report
        st.session_state.cm = cm
        st.session_state.feature_names = feature_names
        st.session_state.model_trained = True
        
    st.sidebar.success(f"✅ Model trained! Accuracy: {acc*100:.2f}%")

# Display model status
if st.session_state.model_trained:
    st.sidebar.metric("Model Accuracy", f"{st.session_state.accuracy*100:.2f}%")
    st.sidebar.success("✅ Model Ready")
else:
    st.sidebar.warning("⚠️ Model not trained yet")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This system uses **Random Forest** machine learning 
algorithm to detect network intrusions in real-time.

**Features:**
- CIC-IDS2017 real dataset
- Real-time predictions
- Performance metrics
- Interactive dashboard
""")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Live Detection", "📈 Model Performance"])

# Tab 1: Dashboard Overview
with tab1:
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Status", 
                  "✅ Active" if st.session_state.model_trained else "⚠️ Inactive")
    
    with col2:
        st.metric("Detection Mode", "Real CIC-IDS2017 Dataset")
    
    with col3:
        st.metric("Total Predictions", len(st.session_state.prediction_history))
    
    st.markdown("---")
    
    st.subheader("System Architecture")
    st.markdown("""
    **Components:**
    1. **Data Layer**: Real CIC-IDS2017 Network Traffic Dataset
    2. **ML Engine**: Random Forest Classifier (100 trees)
    3. **Detection Layer**: Real-time anomaly classification
    4. **Visualization**: Streamlit-based dashboard
    
    **Workflow:**
    - Network packets are analyzed for 10 key features from real traffic
    - Random Forest model classifies traffic as Normal or Attack
    - Results are displayed with confidence scores
    """)

# Tab 2: Live Traffic Simulator
with tab2:
    st.header("🔍 Live Traffic Simulator")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first using the sidebar button.")
    else:
        st.markdown("### Input Network Packet Parameters")
        st.info("💡 These features are based on the CIC-IDS2017 dataset structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            flow_duration = st.slider("Flow Duration (ms)", 0, 120000, 5000)
            total_fwd_packets = st.slider("Forward Packets", 0, 100, 10)
            total_bwd_packets = st.slider("Backward Packets", 0, 100, 10)
            flow_bytes_per_sec = st.slider("Flow Bytes/Sec", 0.0, 1000000.0, 50000.0)
            flow_packets_per_sec = st.slider("Flow Packets/Sec", 0.0, 1000.0, 50.0)
        
        with col2:
            flow_iat_mean = st.slider("Flow IAT Mean", 0.0, 50000.0, 5000.0)
            fwd_packet_length_mean = st.slider("Fwd Packet Length Mean", 0.0, 1500.0, 500.0)
            bwd_packet_length_mean = st.slider("Bwd Packet Length Mean", 0.0, 1500.0, 500.0)
            packet_length_mean = st.slider("Packet Length Mean", 0.0, 1500.0, 500.0)
            avg_packet_size = st.slider("Avg Packet Size", 0.0, 1500.0, 500.0)
        
        if st.button("🔍 Analyze Traffic"):
            features = [
                flow_duration, total_fwd_packets, total_bwd_packets, 
                flow_bytes_per_sec, flow_packets_per_sec, flow_iat_mean,
                fwd_packet_length_mean, bwd_packet_length_mean,
                packet_length_mean, avg_packet_size
            ]
            
            prediction, probability = predict_traffic(
                st.session_state.model, 
                features, 
                st.session_state.feature_names
            )
            
            # Store prediction
            st.session_state.prediction_history.append({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': prediction,
                'confidence': max(probability) * 100
            })
            
            st.markdown("---")
            st.subheader("Detection Result")
            
            if prediction == 1:
                st.error("🚨 **ATTACK DETECTED**")
                st.metric("Confidence", f"{probability[1]*100:.2f}%")
            else:
                st.success("✅ **NORMAL TRAFFIC**")
                st.metric("Confidence", f"{probability[0]*100:.2f}%")
            
            # Display probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Normal Probability", f"{probability[0]*100:.2f}%")
            with col2:
                st.metric("Attack Probability", f"{probability[1]*100:.2f}%")

# Tab 3: Model Performance
with tab3:
    st.header("📈 Model Performance Metrics")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first using the sidebar button.")
    else:
        st.subheader("Classification Report")
        
        # Display accuracy
        st.metric("Overall Accuracy", f"{st.session_state.accuracy*100:.2f}%")
        
        # Classification report
        report_df = pd.DataFrame(st.session_state.report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(st.session_state.cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Feature Importance
        st.subheader("Feature Importance")
        if st.session_state.model_trained:
            importances = st.session_state.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
            plt.title('Feature Importance in Random Forest Model')
            plt.xlabel('Importance Score')
            st.pyplot(fig)
        
        # Prediction History
        if len(st.session_state.prediction_history) > 0:
            st.markdown("---")
            st.subheader("Recent Predictions")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['prediction'] = history_df['prediction'].map({0: 'Normal', 1: 'Attack'})
            st.dataframe(history_df.tail(10))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>AI-Based Network Intrusion Detection System | Built with Streamlit & Scikit-Learn</p>
    <p>Dataset: CIC-IDS2017 (Wednesday Working Hours) | Algorithm: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)
