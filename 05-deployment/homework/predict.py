import pickle
from fastapi import FastAPI, Body
import uvicorn
from pathlib import Path

app = FastAPI(title="customer-churn-prediction")

model_path = Path(__file__).parent / "pipeline_v1.bin"
with open(model_path, 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: dict = Body(...)):
    # The model likely expects a list of feature dicts, not a single dict
    customer_data = [customer]
    prob = predict_single(customer_data)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)