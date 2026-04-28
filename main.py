import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv("data.csv")

# Remove ID column
data = data.drop(columns=["Unnamed"])

# Features and labels
X = data.drop(columns=["y"])
y = data["y"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

sample = X_test.iloc[[0]]   # take one unseen sample
pred = model.predict(sample)[0]

labels = {
    1: "Emergency Brain Event",
    2: "Unusual Brain Pattern",
    3: "Stable Brain Activity",
    4: "Relaxed / Resting",
    5: "Awake / Alert"
}

print("Prediction:", labels[pred])
# Results
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))