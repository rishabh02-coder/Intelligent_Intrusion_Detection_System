import streamlit as st
import pandas as pd
import joblib
import os
import subprocess

st.title("🛡️ Intrusion Detection System ")
st.write("This app uses a trained Random Forest model on the NSL-KDD dataset to detect network intrusions.")

# Run vis.py and capture the output
st.subheader("📜 Backend Logs from main.py")
if st.button("Run Backend and Show Logs"):
    result = subprocess.run(["python", "vis.py"], capture_output=True, text=True)
    st.text(result.stdout)

# Input section for prediction
st.subheader("🔍 Predict Intrusion Type")

# Basic user inputs (you can expand this as needed)
duration = st.number_input("Duration", min_value=0)
protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
service = st.text_input("Service", "http")
flag = st.text_input("Flag", "SF")
src_bytes = st.number_input("Source Bytes", min_value=0)
dst_bytes = st.number_input("Destination Bytes", min_value=0)
count = st.number_input("Connection Count", min_value=0)
srv_count = st.number_input("Service Count", min_value=0)

# Button to trigger prediction
if st.button("Predict Intrusion"):
    try:
        model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Encode protocol_type, service, flag manually (or later use same LabelEncoder logic)
        # For now, hardcoded based on your trained encoders:
        protocol_map = {"icmp": 0, "tcp": 1, "udp": 2}
        flag_map = {"SF": 9}  # Adjust as per your training encoder
        service_map = {"http": 20}  # Adjust as per your training encoder

        input_data = pd.DataFrame([[
            duration, 
            protocol_map.get(protocol_type, 1), 
            service_map.get(service, 20), 
            flag_map.get(flag, 9),
            src_bytes, dst_bytes,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            count, srv_count,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]], columns=[
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files",
            "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
            "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate"
        ])

        # Scale features
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)
        st.success(f"🚨 Predicted Label: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
