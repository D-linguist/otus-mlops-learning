from os import getenv

from datetime import datetime
import pyspark
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window


def convert_timestamp(s):
    s = s.strftime("%Y-%m-%d %H:%M:%S")
    if s[11:13] == '24':
        return s[:11] + '00' + s[13:]
    return s


def is_night(tx_hour):
    is_night = tx_hour <= 6
    return int(is_night)


def preprocess(df):
    
    convert_timestamp_udf = udf(convert_timestamp)
    df = df.withColumn("tx_datetime", convert_timestamp_udf(df["tx_datetime"]))
    df = df.withColumn("ts", to_timestamp(df["tx_datetime"], "yyyy-MM-dd HH:mm:ss"))
    
    df = df.withColumn("year", year(df["ts"]))
    df = df.withColumn("month", month(df["ts"]))
    df = df.withColumn("day", dayofmonth(df["ts"]))
    df = df.withColumn("is_weekend", dayofweek("ts").isin([1,7]).cast("int"))
    is_night_udf = udf(is_night, IntegerType())
    df = df.withColumn("is_night", is_night_udf(hour("ts")))
    
    return df


def get_features(df, windows_size_in_days=[1,7,30]):
    delay = 7
    precision_loss = 1

    for window_size in windows_size_in_days:

        window = Window \
            .partitionBy("customer_id") \
            .orderBy(col("ts").cast("long")) \
            .rangeBetween(-window_size * 86400, 0)  # seconds in a day

        df = df \
            .withColumn('nb_tx_window', count("tx_amount").over(window)) \
            .withColumn('avg_amount_tx_window', avg("tx_amount").over(window))

        df = df \
            .withColumnRenamed('nb_tx_window', 'customer_id_nb_tx_' + str(window_size)) \
            .withColumnRenamed('avg_amount_tx_window', 'customer_id_avg_amount_' + str(window_size))
        
    precision_loss = 3600
    df = df.withColumn("truncated_ts", (col("ts").cast("long") / precision_loss).cast("long"))

    delay_window = Window \
        .partitionBy("terminal_id") \
        .orderBy(col("truncated_ts").cast("long")) \
        .rangeBetween(-delay * 86400 // precision_loss, 0)  

    df = df \
        .withColumn('nb_fraud_delay', sum("tx_fraud").over(delay_window)) \
        .withColumn('nb_tx_delay', count("tx_fraud").over(delay_window)) 

    for window_size in windows_size_in_days:

        delay_window_size = Window \
            .partitionBy("terminal_id") \
            .orderBy(col("truncated_ts").cast("long")) \
            .rangeBetween(-(delay + window_size) * 86400 // precision_loss, 0)

        df = df \
            .withColumn('nb_fraud_delay_window', sum("tx_fraud").over(delay_window_size)) \
            .withColumn('nb_tx_delay_window', count("tx_fraud").over(delay_window_size)) 

        df = df \
            .withColumn('nb_fraud_window', col('nb_fraud_delay_window') - col('nb_fraud_delay')) \
            .withColumn('nb_tx_window', col('nb_tx_delay_window') - col('nb_tx_delay')) 

        df = df \
            .withColumn('risk_window', when(col('nb_tx_window') > 0, col('nb_fraud_window') / col('nb_tx_window')).otherwise(0))

        df = df \
            .withColumnRenamed('nb_tx_window', 'terminal_id_nb_tx_' + str(window_size)) \
            .withColumnRenamed('risk_window', 'terminal_id_risk_' + str(window_size)) 

    return df



def build_train_pipeline():
    stages = []

    numerical_columns = [
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
         "terminal_id_risk_30"
    ]
    
    assembler_input = [column for column in numerical_columns] 
    vector_assembler = VectorAssembler(inputCols=assembler_input, outputCol="features")
    stages += [vector_assembler]
    
    classification = LogisticRegression(featuresCol='features', labelCol='tx_fraud')
    stages += [classification]

    pipeline = Pipeline(stages=stages)
    
    return pipeline


def calculate_accuracy(predictions):
    predictions = predictions.withColumn(
        "fraudPrediction",
        when((predictions.tx_fraud==1) & (predictions.prediction==1), 1).otherwise(0)
    )

    accurateFraud = predictions.groupBy("fraudPrediction").count().where(predictions.fraudPrediction==1).head()[1]
    totalFraud = predictions.groupBy("tx_fraud").count().where(predictions.tx_fraud==1).head()[1]
    accuracy = (accurateFraud/totalFraud)*100
    return accuracy


if __name__ == "__main__":
    print(f'MLFLOW_TRACKING_URI:  {getenv("MLFLOW_TRACKING_URI")}')
    
    spark = (
        pyspark.sql.SparkSession.builder
            .config("spark.executor.cores", 4)
            .appName("fraud_data_train")
            .getOrCreate()
    )
    
    df = spark.read.parquet("s3a://object-storage-3/fraud_data/")

    print("Preprocessing...")
    df = preprocess(df)
    
    df = df.orderBy('ts')
    print("Getting features...")
    df = get_features(df, windows_size_in_days=[1, 7, 30])

    print("=" * 50)
    print("=" * 50)

    df.select(min('tx_datetime').alias('min_date')).show()
    df.select(max('tx_datetime').alias('max_date')).show()

    df.select(min('tx_time_days')).show()
    df.select(max('tx_time_days')).show()

    print("=" * 50)
    print("=" * 50)

    df_train = df.filter(col('ts').between(datetime.strptime("2022-10-05", "%Y-%m-%d"), datetime.strptime("2022-11-04", "%Y-%m-%d")))
    df_test = df.filter(col('ts').between(datetime.strptime("2022-11-05", "%Y-%m-%d"), datetime.strptime("2022-12-04", "%Y-%m-%d")))

    client = MlflowClient()
    experiment = client.get_experiment_by_name("Fraud")
    experiment_id = experiment.experiment_id

    run_name = 'Fraud_data_pipeline' + ' ' + str(datetime.now())

    print(f'MLFLOW_TRACKING_URI:  {getenv("MLFLOW_TRACKING_URI")}')

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):

        print("Building...")

        pipeline = build_train_pipeline()
        model = pipeline.fit(df_train)

        print("Evaluation...")
        evaluator = BinaryClassificationEvaluator(labelCol='tx_fraud', rawPredictionCol='prediction')

        predictions_train = model.transform(df_train)
        predictions_train.head()
        areaUnderROC_train = evaluator.evaluate(predictions_train)

        predictions_test = model.transform(df_test)
        areaUnderROC_test = evaluator.evaluate(predictions_test)

        run_id = mlflow.active_run().info.run_id
        print(f"Logging metrics to MLflow run {run_id} ...")
        mlflow.log_metric("ROC-train", areaUnderROC_train)
        print(f"Model ROC-train: {areaUnderROC_train}")
        mlflow.log_metric("ROC-test", areaUnderROC_test)
        print(f"Model ROC-test: {areaUnderROC_test}")

        print("Saving model locally...")
        model.write().overwrite().save("./models/latest.mdl")

        FraudPredictionAccuracy = calculate_accuracy(predictions_train)
        print("FraudPredictionAccuracy train:", FraudPredictionAccuracy)
        FraudPredictionAccuracy = calculate_accuracy(predictions_test)
        print("FraudPredictionAccuracy test:", FraudPredictionAccuracy)

        print("Exporting/logging model ...")
        mlflow.spark.log_model(model, 'fraud_classifier', registered_model_name='fraud_classifier')
        print("Done")
    
    spark.stop()