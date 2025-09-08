# (paste train_model.py content provided below)
# backend/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/all_assets.csv")
X = df[["momentum", "volatility", "pe_ratio", "sector_signal", "liquidity"]]
y = df["action"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=7)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification report for model:")
print(classification_report(y_test, y_pred))

joblib.dump(clf, "models/invest_model_rf.joblib")
print("Saved model to models/invest_model_rf.joblib")
