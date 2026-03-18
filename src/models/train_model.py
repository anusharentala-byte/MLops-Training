import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("file:./mlruns")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.data.load_data import load_data
from src.features.build_features import build_features


def train():

    # Load dataset
    data = load_data("data/raw/student_data.csv")

    # Feature split
    X, y = build_features(data)

    # Parameters
    test_size = 0.2
    random_state = 42

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Start MLflow experiment
    with mlflow.start_run():

        model = LinearRegression()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        # Log parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        mlflow.log_metric("mse", mse)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        joblib.dump(model, "models/model.pkl")

        print("Model trained successfully!")
        print("MSE:", mse)


if __name__ == "__main__":
    train()