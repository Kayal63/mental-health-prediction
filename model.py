import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("dataset/mental_health_data.csv")

# Preprocess
df = df.drop(columns=['user_id'], errors='ignore')
df['mental_risk'] = df['mental_risk'].map({'yes': 1, 'no': 0})

X = df.drop(columns=['mental_risk'])
y = df['mental_risk']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

