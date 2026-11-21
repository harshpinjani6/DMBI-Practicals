import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Experiment 5: Classification using Decision Tree and Naive Bayes
# Data transcribed from the 20-record student performance table in the DOCX.
df = pd.read_csv("student_performance_exp05.csv")
print("First five records:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df["Result"].value_counts())

X = df[["Attendance", "Study_Hours", "Internal_Marks"]]
y = df["Result"]
encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]] * 3)
X_encoded = encoder.fit_transform(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.25, random_state=42,
    stratify=y_encoded
)
print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

def evaluate(name, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print("Actual:", label_encoder.inverse_transform(y_test))
    print("Predicted:", label_encoder.inverse_transform(predictions))
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions,
          target_names=label_encoder.classes_, zero_division=0))
    return predictions

dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_predictions = evaluate("Decision Tree", dt_model)

plt.figure(figsize=(14, 8))
plot_tree(dt_model, feature_names=list(X.columns),
          class_names=list(label_encoder.classes_), filled=True, rounded=True)
plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.show()

nb_model = CategoricalNB()
nb_predictions = evaluate("Naive Bayes", nb_model)

print("\nFINAL COMPARISON")
for name, predictions in [("Decision Tree", dt_predictions),
                          ("Naive Bayes", nb_predictions)]:
    correct = int((predictions == y_test).sum())
    print(name, "- Correct:", correct,
          "Incorrect:", len(y_test) - correct,
          "Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
