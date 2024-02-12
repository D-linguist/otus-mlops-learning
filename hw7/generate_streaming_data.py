from os import getenv
from dotenv import load_dotenv

import json
from typing import Dict, NamedTuple
import logging
import random
import datetime
import argparse
from collections import namedtuple

import kafka
from confluent_kafka import Producer


import pyspark
from pyspark.sql.functions import col, to_json, struct

load_dotenv()
KAFKA_URL = getenv("KAFKA_URL")
TOPIC = getenv("TOPIC_0")
USER = getenv("KUSER")
PASS = getenv("KPASS")


def main():
    print(50*'=')
    print(f'KAFKA_UURL {KAFKA_URL}')
    print(f'TOPIC {TOPIC}')
    print(50*'=')
    
    spark = (
        pyspark.sql.SparkSession.builder
            .appName("kafka_generate")
            .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    
    print("Generating df")
    
    df = spark.read.parquet("s3a://object-storage-3/fraud_data_prep/part-00000-6ecb573f-949c-4160-9b5e-6f03226d372a-c000.snappy.parquet")
    
    df_s = df.sample(10/df.count()).limit(10)
   
    
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
        "tx_fraud"
    ]
    df_s = df_s.select(columns)

    df_kafka = df_s.select(
        col("transaction_id").cast("string"),
        to_json(struct([df_s[x] for x in df_s.columns]))
    ).toDF("key", "value")
    
    
    print("Connecting to Kafka")
    
    options = {
        "kafka.sasl.mechanism": "SCRAM-SHA-512",
        "kafka.security.protocol" : "SASL_PLAINTEXT",
        "kafka.bootstrap.servers": f'{KAFKA_URL}:9092',
        "topic": TOPIC,
        "kafka.sasl.jaas.config": f'org.apache.kafka.common.security.scram.ScramLoginModule required username="{USER}" password="{PASS}";',
        "group.id": 'test',
    }
  
    (df_kafka
        .write
        .format("kafka")
        .options(**options)
        .save()
    )


if __name__ == "__main__":
    main()
