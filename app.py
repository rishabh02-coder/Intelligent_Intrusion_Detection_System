import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔐 Intrusion Detection System")
st.markdown("Enter the network connection features below to predict if it’s a normal or an attack connection.")

# Input fields for features
features = {
    "duration": st.number_input("Duration", min_value=0),
    "protocol_type": st.selectbox("Protocol Type", options=["tcp", "udp", "icmp"]),
    "service": st.text_input("Service (e.g., http, ftp, smtp)", value="http"),
    "flag": st.text_input("Flag (e.g., SF, S0)", value="SF"),
    "src_bytes": st.number_input("Source Bytes", min_value=0),
    "dst_bytes": st.number_input("Destination Bytes", min_value=0),
    "land": st.selectbox("Land", options=[0, 1]),
    "wrong_fragment": st.number_input("Wrong Fragment", min_value=0),
    "urgent": st.number_input("Urgent", min_value=0),
    "hot": st.number_input("Hot", min_value=0),
    "num_failed_logins": st.number_input("Failed Logins", min_value=0),
    "logged_in": st.selectbox("Logged In", options=[0, 1]),
    "num_compromised": st.number_input("Compromised Count", min_value=0),
    "root_shell": st.number_input("Root Shell", min_value=0),
    "su_attempted": st.number_input("SU Attempted", min_value=0),
    "num_root": st.number_input("Num Root", min_value=0),
    "num_file_creations": st.number_input("File Creations", min_value=0),
    "num_shells": st.number_input("Shells", min_value=0),
    "num_access_files": st.number_input("Access Files", min_value=0),
    "num_outbound_cmds": st.number_input("Outbound Commands", min_value=0),
    "is_host_login": st.selectbox("Is Host Login", options=[0, 1]),
    "is_guest_login": st.selectbox("Is Guest Login", options=[0, 1]),
    "count": st.number_input("Count", min_value=0),
    "srv_count": st.number_input("SRV Count", min_value=0),
    "serror_rate": st.number_input("Serror Rate", min_value=0.0),
    "srv_serror_rate": st.number_input("SRV Serror Rate", min_value=0.0),
    "rerror_rate": st.number_input("Rerror Rate", min_value=0.0),
    "srv_rerror_rate": st.number_input("SRV Rerror Rate", min_value=0.0),
    "same_srv_rate": st.number_input("Same SRV Rate", min_value=0.0),
    "diff_srv_rate": st.number_input("Diff SRV Rate", min_value=0.0),
    "srv_diff_host_rate": st.number_input("SRV Diff Host Rate", min_value=0.0),
    "dst_host_count": st.number_input("Dst Host Count", min_value=0),
    "dst_host_srv_count": st.number_input("Dst Host SRV Count", min_value=0),
    "dst_host_same_srv_rate": st.number_input("Dst Host Same SRV Rate", min_value=0.0),
    "dst_host_diff_srv_rate": st.number_input("Dst Host Diff SRV Rate", min_value=0.0),
    "dst_host_same_src_port_rate": st.number_input("Dst Host Same Src Port Rate", min_value=0.0),
    "dst_host_srv_diff_host_rate": st.number_input("Dst Host SRV Diff Host Rate", min_value=0.0),
    "dst_host_serror_rate": st.number_input("Dst Host Serror Rate", min_value=0.0),
    "dst_host_srv_serror_rate": st.number_input("Dst Host SRV Serror Rate", min_value=0.0),
    "dst_host_rerror_rate": st.number_input("Dst Host Rerror Rate", min_value=0.0),
    "dst_host_srv_rerror_rate": st.number_input("Dst Host SRV Rerror Rate", min_value=0.0),
}

# Add label and difficulty fields (for info/logging — not used in prediction)
label = st.selectbox("Label (actual class)", options=[
    "normal", "neptune", "smurf", "satan", "ipsweep", "portsweep", "nmap",
    "back", "teardrop", "warezclient", "warezmaster", "pod", "guess_passwd",
    "buffer_overflow", "imap", "rootkit", "perl", "loadmodule", "ftp_write",
    "multihop", "phf", "spy", "land"
])

difficulty = st.selectbox("Difficulty Level (optional)", options=list(range(0,101)))

# Categorical encoding maps
protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
flag_map = {"SF": 9, "S0": 6}
service_map = {"http": 20, "ftp": 4, "smtp": 21}

# Replace string inputs with encoded values
encoded_input = []
for key in features:
    if key == "protocol_type":
        encoded_input.append(protocol_map.get(features[key], 0))
    elif key == "service":
        encoded_input.append(service_map.get(features[key], 0))
    elif key == "flag":
        encoded_input.append(flag_map.get(features[key], 0))
    else:
        encoded_input.append(features[key])

# Predict button
if st.button("Predict"):
    input_array = np.array(encoded_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    if prediction == "normal":
        st.success("✅ This connection is **Normal**.")
    else:
        st.error("⚠️ This connection is **Malicious** (Attack detected).")

    st.markdown("---")
    st.info(f"🔍 **Provided Label:** `{label}`")
    st.info(f"📊 **Difficulty Level:** `{difficulty}`")
