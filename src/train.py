import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from pathlib import Path

# Load dataset
data_path = Path("data") / "BostonHousing.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop("medv", axis=1)
y = df["medv"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training complete.")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save trained model
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "linear_regression_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {model_path}")