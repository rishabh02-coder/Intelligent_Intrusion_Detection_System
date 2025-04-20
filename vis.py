import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Define column names for the NSL-KDD dataset (42 columns)
columns = [
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
    "dst_host_srv_rerror_rate", "label"
]

# Load training and testing data (fix: handle extra column)
try:
    train_df = pd.read_csv("KDDTrain+.txt", names=columns + ["extra"], usecols=columns)
    test_df = pd.read_csv("KDDTest+.txt", names=columns + ["extra"], usecols=columns)
except FileNotFoundError:
    print("Error: Dataset files not found. Ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are in the correct directory.")
    exit()

# Strip whitespace from object columns
for df in [train_df, test_df]:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

print("Original Data Sample:")
print(train_df.head())

# Handle missing values
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

# Encode categorical columns
categorical_cols = ["protocol_type", "service", "flag", "label"]
for col in categorical_cols:
    if col in train_df.columns:
        combined_data = pd.concat([train_df[col], test_df[col]], axis=0)
        encoder = LabelEncoder()
        encoder.fit(combined_data)
        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])
    else:
        print(f"Warning: Column '{col}' not found in dataset!")

print("\nEncoding Done! Sample Data:")
print(train_df.head())

# Select numerical columns (excluding the label)
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "label" in num_cols:
    num_cols.remove("label")

# Apply StandardScaler to numerical features
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print("\nNormalization Done! Sample Data:")
print(train_df.head())

# Prepare training and testing datasets
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Debug check: make sure no non-numeric columns remain
non_numeric_columns = X_train.select_dtypes(include=["object"]).columns
if not non_numeric_columns.empty:
    print("\n🚨 Non-numeric columns found in X_train:")
    print(non_numeric_columns)
    print("\nSample values from those columns:")
    for col in non_numeric_columns:
        print(f"{col}: {X_train[col].unique()[:5]}")
    exit()

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Save model and scaler for frontend use
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")
