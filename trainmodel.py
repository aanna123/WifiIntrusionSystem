import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import joblib  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  

# Load Dataset  
df = pd.read_csv("../data/UNSW_NB15.csv")  
print("Dataset Loaded Successfully!")

# Preprocessing  
df = df.dropna()  # Remove missing values  

# Encoding Categorical Variables  
label_encoder = LabelEncoder()  
df['attack_cat'] = label_encoder.fit_transform(df['attack_cat'])  
df['proto'] = label_encoder.fit_transform(df['proto'])  
df['service'] = label_encoder.fit_transform(df['service'])  
df['state'] = label_encoder.fit_transform(df['state'])  

# Splitting Data  
X = df.drop(["label", "attack_cat"], axis=1)  
y = df["label"]  # 1 = Intrusion, 0 = Normal  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Feature Scaling  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Train Model  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train_scaled, y_train)  
print("Model Training Completed!")

# Evaluate Model  
y_pred = model.predict(X_test_scaled)  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy * 100:.2f}%")  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  

# Save Model  
joblib.dump(model, "model.pkl")  
print("Model Saved as model.pkl!")  

# Confusion Matrix Plot  
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.title("Confusion Matrix")  
plt.show()
