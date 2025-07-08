import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib

# Load CSV
df = pd.read_csv("fraud_summary.csv")

# Optional: Create output folder
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

### --- EDA SECTION --- ###
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Class Distribution (is_fraud) ---")
print((df['Sum_actual_fraud'] > 0).astype(int).value_counts())

print("\n--- Summary Statistics ---")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{plot_dir}/correlation_matrix.png")
plt.close()

# Distribution plots
numeric_cols = ['Sum_flag_rule', 'Sum_actual_fraud', 'Sum_transaction_amount']
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/distribution_{col}.png")
    plt.close()

# Boxplot grouped by fraud
df['is_fraud'] = (df['Sum_actual_fraud'] > 0).astype(int)
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='is_fraud', y=col, data=df)
    plt.title(f"{col} vs Fraud")
    plt.xlabel("Is Fraud")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/boxplot_{col}_vs_fraud.png")
    plt.close()

### --- MODELING SECTION --- ###
# Backup company_id for future use
company_ids = df['company_id']

# Drop non-feature columns
df = df.drop(['company_id', 'risk_level', 'percent_fraud'], axis=1)

# Features and Target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{plot_dir}/confusion_matrix.png")
plt.close()

# Save features for app
X['company_id'] = company_ids
X.set_index('company_id').to_csv('fraud_app/company_features.csv')

# Save trained model
joblib.dump(model, 'fraud_app/fraud_model.pkl')
print(" Model and features saved.")
