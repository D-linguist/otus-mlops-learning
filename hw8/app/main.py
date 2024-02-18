
import uvicorn
import mlflow
import pandas as pd

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter


from pydantic import BaseModel
from datetime import datetime

class Transaction(BaseModel):
    transaction_id: int
    ts: datetime
    tx_amount: float
    is_weekend: int
    is_night: int
    customer_id_nb_tx_1: int
    customer_id_avg_amount_1: float
    customer_id_nb_tx_7: int
    customer_id_avg_amount_7: float
    customer_id_nb_tx_30: int
    customer_id_avg_amount_30: float
    terminal_id_nb_tx_1: int
    terminal_id_risk_1: float
    terminal_id_nb_tx_7: int
    terminal_id_risk_7: float
    terminal_id_nb_tx_30: int
    terminal_id_risk_30: float
    tx_fraud: int


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

COUNTER = Counter("fraud", "Fraud predicted")


class Model:
    pipeline:  None


@app.on_event("startup")
def load_model():
    Model.pipeline = mlflow.pyfunc.load_model(model_uri=f"models:/fraud_classifier/2")


@app.post("/predict/")
async def predict(transaction: Transaction):
    df = pd.DataFrame(jsonable_encoder(transaction), index=[0])
    res = Model.pipeline.predict(df)
    if int(res[0]) == 1:
        COUNTER.inc()
    print("Prediction:", res)
 
    return {"prediction": res}


@app.get("/")
async def root():
    return "Hello World!"


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0", reload=True)