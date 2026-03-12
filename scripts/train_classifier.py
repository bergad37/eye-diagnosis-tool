import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

# -------------------
# Load features
# -------------------
data = np.load("data/aptos_features.npz")
X = data["X"]
y = data["y"]

print("Features:", X.shape)
print("Labels:", y.shape)

# -------------------
# Train / validation split
# -------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------
# Compute class weights
# -------------------
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weights_dict)

# -------------------
# Scale features
# -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# -------------------
# Train classifier with class weights
# -------------------
clf = LogisticRegression(
    max_iter=5000,           # Increase iterations for convergence
    solver="lbfgs",          
    class_weight=class_weights_dict
)

clf.fit(X_train, y_train)

# -------------------
# Evaluate
# -------------------
preds = clf.predict(X_val)
print(classification_report(y_val, preds))

# -------------------
# Save trained model and scaler
# -------------------
joblib.dump(clf, "models/aptos_classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Trained model saved to 'models/aptos_classifier.pkl'")
print("Scaler saved to 'models/scaler.pkl'")
