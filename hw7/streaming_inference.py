from os import getenv
from dotenv import load_dotenv


load_dotenv()
KAFKA_URL = getenv("KAFKA_URL")
TOPIC_READ = getenv("TOPIC_0")
TOPIC_WRITE = getenv("TOPIC_1")
USER = getenv("KUSER")
PASS = getenv("KPASS")

import pyspark
import mlflow
from pyspark.sql.functions import from_json, col, to_json, struct
from pyspark.sql.types import StructType, TimestampType, IntegerType, StructField, DoubleType



def main():
    spark = (
        pyspark.sql.SparkSession.builder
            .appName("stream_inference")
            .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    model = mlflow.spark.load_model(model_uri=f"models:/fraud_classifier/2")
    
    schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("ts", TimestampType(), True),
        StructField("tx_amount", DoubleType(), True),
        StructField("is_weekend", IntegerType(), True),
        StructField("is_night", IntegerType(), True),
        StructField("customer_id_nb_tx_1", IntegerType(), True),
        StructField("customer_id_avg_amount_1", DoubleType(), True),
        StructField("customer_id_nb_tx_7", IntegerType(), True),
        StructField("customer_id_avg_amount_7", DoubleType(), True),
        StructField("customer_id_nb_tx_30", IntegerType(), True),
        StructField("customer_id_avg_amount_30", DoubleType(), True),
        StructField("terminal_id_nb_tx_1", IntegerType(), True),
        StructField("terminal_id_risk_1", DoubleType(), True),
        StructField("terminal_id_nb_tx_7", IntegerType(), True),
        StructField("terminal_id_risk_7", DoubleType(), True),
        StructField("terminal_id_nb_tx_30", IntegerType(), True),
        StructField("terminal_id_risk_30", DoubleType(), True),
        StructField("tx_fraud", IntegerType(), True),
    ])

    
    options_0 = {
        "kafka.sasl.mechanism": "SCRAM-SHA-512",
        "kafka.security.protocol" : "SASL_PLAINTEXT",
        "kafka.bootstrap.servers": f'{KAFKA_URL}:9092',
        "ssl.client.auth": "none",
        "subscribe": TOPIC_READ,
        "kafka.sasl.jaas.config": f'org.apache.kafka.common.security.scram.ScramLoginModule required username="{USER}" password="{PASS}";',
        "group.id": 'test',
        "failOnDataLoss": "false",
    }
    
    
    df_0 = (
        spark.readStream
            .format("kafka")
            .options(**options_0)
            .load()
            .selectExpr("CAST(value AS STRING) as json")
            .select(from_json(col("json"), schema).alias("data"))
            .select("data.*")
    )

    df_0.printSchema()
    
    results = model.transform(df_0)

    
    options_1 = {
        "kafka.sasl.mechanism": "SCRAM-SHA-512",
        "kafka.security.protocol" : "SASL_PLAINTEXT",
        "kafka.bootstrap.servers": f'{KAFKA_URL}:9092',
        "topic": TOPIC_WRITE,
        "kafka.sasl.jaas.config": f' org.apache.kafka.common.security.scram.ScramLoginModule required username="{USER}" password="{PASS}";',
        "checkpointLocation": "/tmp/checkpoint",
    }
    
    (results
        .select(
            col("transaction_id").cast("string"),
            to_json(struct([col(x) for x in ["transaction_id", "ts", "tx_fraud", "probability", "prediction"]]))
        ).toDF("key", "value")
        .writeStream
        .format("kafka")
        .options(**options_1)
        .start()
        .awaitTermination()
    )
    


if __name__ == "__main__":
    main()
