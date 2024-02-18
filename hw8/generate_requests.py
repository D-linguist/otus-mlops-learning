import json
import requests
import time

from dotenv import load_dotenv
from os import getenv

import pandas as pd


load_dotenv()

KEY_ID = getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = getenv("AWS_SECRET_ACCESS_KEY")
REGION = getenv('AWS_REGION')
S3_URL = getenv('MLFLOW_S3_ENDPOINT_URL')
ML_API_URL = getenv('ML_API_URL', 'http://127.0.0.1:8000/predict')


parquet = "s3a://object-storage-3/fraud_data_prep/part-00000-6ecb573f-949c-4160-9b5e-6f03226d372a-c000.snappy.parquet"

df = pd.read_parquet(
    parquet,
    storage_options={
        "key"          : KEY_ID,
        "secret"       : SECRET_KEY,
        "client_kwargs": {
            'verify'      : True,
            'region_name' : REGION,
            'endpoint_url': S3_URL,
        }
    }
)

df = df[df['tx_fraud'] == 1]

for index, row in df.iterrows():
    columns = [
        "transaction_id",
        "ts",
        "tx_amount",
        "is_weekend",
        "is_night",
        "customer_id_nb_tx_1",
        "customer_id_avg_amount_1",
        "customer_id_nb_tx_7",
        "customer_id_avg_amount_7",
        "customer_id_nb_tx_30",
        "customer_id_avg_amount_30",
        "terminal_id_nb_tx_1",
        "terminal_id_risk_1",
        "terminal_id_nb_tx_7",
        "terminal_id_risk_7",
        "terminal_id_nb_tx_30",
        "terminal_id_risk_30",
        "tx_fraud",
    ]
    
    row['ts'] = row['ts'].strftime('%Y-%m-%d %H:%M:%S')

    response = requests.post(
        url=ML_API_URL,
        json=row[columns].to_dict(),
    )
    try:
        print("Prediction:", json.loads(response.text)['prediction'])
    except Exception as e:
        print("Error", e, response.text)

    time.sleep(0.5)