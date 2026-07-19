# Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
print("=" * 60)
print("      DATA CLASSIFICATION USING AI")
print("=" * 60)

iris = load_iris()

# Convert dataset into DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

print("\nDataset Loaded Successfully!\n")

# -----------------------------
# Display Dataset
# -----------------------------
print("First 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Features and Target
# -----------------------------
X = iris.data
y = iris.target

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data :", len(X_test))

# -----------------------------
# Train Model
# -----------------------------
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy*100:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# -----------------------------
# User Prediction
# -----------------------------
print("\n" + "=" * 60)
print("Predict Iris Flower")
print("=" * 60)

try:
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))

    prediction = model.predict([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    flower = iris.target_names[prediction[0]]

    print("\nPrediction:")
    print("Flower Species =", flower)

except ValueError:
    print("\nInvalid Input! Please enter numeric values only.")

print("\nProject Completed Successfully!")