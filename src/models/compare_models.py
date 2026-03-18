import mlflow
import mlflow.sklearn
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.data.load_data import load_data
from src.features.build_features import build_features

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Student Score Prediction")


def train_model(model, model_name, X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} MSE:", mse)


def main():

    data = load_data("data/raw/student_data.csv")

    X, y = build_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor()
    }

    for name, model in models.items():
        train_model(model, name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()