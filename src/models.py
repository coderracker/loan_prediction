from train_pipeline import build_model
from preprocess import prepare_data
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

X_train, X_test, X_all, y_train, y_test = prepare_data()
model = build_model()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/loan_model.joblib")
