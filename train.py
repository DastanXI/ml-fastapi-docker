"""Train a simple classifier on the Iris dataset and save it to model.joblib.

Run:  python train.py
"""
import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.joblib"
RANDOM_STATE = 42


def main() -> None:
    # 1. Load a classic dataset (built into scikit-learn, no download needed)
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = [str(name) for name in iris.target_names]  # setosa, versicolor, virginica

    # 2. Hold out 20% of the data to measure quality on unseen examples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Train a classification model
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. Save the trained model as an artifact that the API will load.
    #    We store the model together with the metadata the API needs.
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
        "accuracy": float(accuracy),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
