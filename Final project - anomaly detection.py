import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

st.title("Linguistic Anomaly Detector")
file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file, encoding='latin1')
    col = st.selectbox("Select Text Column", df.columns)
    contam = st.slider("Contamination", 0.01, 0.20, 0.05)

    if st.button("Detect Anomalies"):
        # Vectorize & Model
        X = TfidfVectorizer(stop_words='english', max_features=500).fit_transform(df[col].fillna(''))
        df['is_anomaly'] = IsolationForest(contamination=contam, random_state=42).fit_predict(X.toarray())
        
        # Filter & Display
        anomalies = df[df['is_anomaly'] == -1]
        st.metric("Anomalies Found", len(anomalies))
        st.dataframe(anomalies[[col]])
        
        # Download Option
        st.download_button("Download Results", anomalies.to_csv(index=False), "anomalies.csv")
