import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

data = pd.DataFrame({
    "hours_studied": [6]
})

prediction = model.predict(data)

print("Predicted score:", prediction[0])