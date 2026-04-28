import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("EEG_data.csv")

# Personalized user
subject_id = 1
df = df[df["SubjectID"] == subject_id].reset_index(drop=True)

# ======================
# WINDOWING
# ======================
window_size = 10   # 10 rows = 5 sec

raw_values = df["Raw"].values
labels = df["user-definedlabeln"].values.astype(int)

X = []
y = []

for i in range(len(df) - window_size):
    window = raw_values[i:i+window_size]
    target = labels[i + window_size - 1]

    X.append(window)
    y.append(target)

X = np.array(X)
y = np.array(y)

# ======================
# SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ======================
# SCALE
# ======================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.reshape(-1,1)).reshape(X_train.shape)
X_test  = scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)

# reshape for deep learning
X_train = X_train.reshape(X_train.shape[0], window_size, 1)
X_test  = X_test.reshape(X_test.shape[0], window_size, 1)

# ======================
# MODEL
# ======================
model = Sequential([
    Conv1D(32, 3, activation="relu", input_shape=(window_size,1)),
    MaxPooling1D(pool_size=2),

    LSTM(32),

    Dense(32, activation="relu"),
    Dropout(0.3),

    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# TRAIN
# ======================
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    verbose=1
)

# ======================
# TEST
# ======================
probs = model.predict(X_test)
pred = np.argmax(probs, axis=1)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))