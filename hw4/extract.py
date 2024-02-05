# import findspark
# findspark.init()

import pyspark as sprk
import sys

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from datetime import datetime as dt


def main():
    print("Starting...")
    session = (
        SparkSession.builder
        .config(conf=SparkConf().setMaster("local[3]"))
        .appName("local")
        .getOrCreate()
    )
    
    print("Session created...")
    session.conf.set('spark.sql.repl.eagerEval.enabled', True) 
    session.conf.set('spark.local.dir', '/home/')
    
    schema = StructType([
        StructField("transaction_id", LongType(), True),
        StructField("tx_datetime", TimestampType(), True),         
        StructField("customer_id", LongType(), True),         
        StructField("terminal_id", LongType(), True),
        StructField("tx_amount", DoubleType(), True),
        StructField("tx_time_seconds", LongType(), True),
        StructField("tx_time_days", LongType(), True),
        StructField("tx_fraud", LongType(), True),
        StructField("tx_fraud_scenario", LongType(), True),
    ])
    
    print(f"{dt.now()} Connecting to the bucket...")
    data = session.read.schema(schema).option("comment", "#").csv("s3a://object-storage-1/", header=False)
    
    print(f"{dt.now()} Clearing the data...")
    cleared_data = data.na.drop()\
    .filter(col("tx_amount") > 0)\
    .filter(col("customer_id") > 0)\
    .dropDuplicates(['transaction_id'])
    
    print(f"{dt.now()} Uploading data...")
    cleared_data.write.parquet("s3a://object-storage-2/parquet/fraud_data.parquet", mode="overwrite")
    
    print(f"{dt.now()} Finished...")


if __name__ == "__main__":
    main()