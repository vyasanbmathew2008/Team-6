from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
DATA = BASE / "data" / "raw"
MODELS = BASE / "models"

DATASETS = {
    "heart": ("heart_disease.csv", "Heart Disease Status"),
    "diabetes": ("diabetes_dataset.csv", "Target"),
    "infection": ("health_dataset.csv", "Disease"),
}

def train_model(name, file_name, target):
    path = DATA / file_name
    if not path.exists():
        print("Missing:", path)
        return

    df = pd.read_csv(path).drop_duplicates()
    if target not in df.columns:
        print("Target not found:", target)
        return

    df = df.dropna(subset=[target])
    X = pd.get_dummies(df.drop(columns=[target]), dummy_na=True)
    y = df[target].astype(str)

    if y.nunique() < 2:
        print("Not enough classes:", name)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print(name, "accuracy:", round(model.score(X_test, y_test), 3))

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / f"{name}.pkl", "wb") as file:
        pickle.dump({
            "model": model,
            "columns": X.columns.tolist(),
            "target": target
        }, file)

def main():
    for name, values in DATASETS.items():
        train_model(name, *values)

if __name__ == "__main__":
    main()
