# src/services/streaming/spark_streaming_processor.py
import os
import sys

# ADDED: Add project root to sys.path for local module imports
sys.path.append('/app')
sys.path.append('/app/src')

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, hour, length, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, TimestampType
from dotenv import load_dotenv
from src.utils.schema_definitions import raw_call_schema

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw-calls")
PARQUET_RAW_PATH = os.getenv("PARQUET_RAW_PATH", "/data/data_lake/raw_calls_parquet")
PARQUET_PROCESSED_PATH = os.getenv("PARQUET_PROCESSED_PATH", "/data/data_lake/processed_calls_parquet")
CHECKPOINT_BASE_PATH = os.getenv("CHECKPOINT_BASE_PATH", "/tmp/spark_checkpoints")


class SparkStreamingProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("CallCenterStreamingProcessor") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
            .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_BASE_PATH) \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        print("SparkSession created for streaming processing.")

    def process_stream(self):
        kafka_df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
            .option("subscribe", KAFKA_TOPIC) \
            .load()

        parsed_df = kafka_df.select(
            from_json(col("value").cast("string"), raw_call_schema).alias("data"),
            col("timestamp").alias("kafka_ingest_time")
        ).select("data.*", "kafka_ingest_time")

        processed_df = parsed_df \
            .withColumn("timestamp", to_timestamp(col("timestamp"))) \
            .withColumn("processing_time", to_timestamp(col("kafka_ingest_time"))) \
            .withColumn("hour_of_day", hour(col("timestamp"))) \
            .withColumn("transcript_length", length("transcript"))

        query_raw_parquet = processed_df \
            .writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", PARQUET_RAW_PATH) \
            .option("checkpointLocation", os.path.join(CHECKPOINT_BASE_PATH, "raw_parquet")) \
            .partitionBy("hour_of_day") \
            .trigger(processingTime='30 seconds') \
            .start()
        print(f"Streaming raw data to Parquet at {PARQUET_RAW_PATH}")

        query_processed_parquet = processed_df \
            .writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", PARQUET_PROCESSED_PATH) \
            .option("checkpointLocation", os.path.join(CHECKPOINT_BASE_PATH, "processed_parquet")) \
            .partitionBy("product_category", "hour_of_day") \
            .trigger(processingTime='30 seconds') \
            .start()
        print(f"Streaming processed data to Parquet at {PARQUET_PROCESSED_PATH}")

        return query_raw_parquet, query_processed_parquet


if __name__ == "__main__":
    processor = SparkStreamingProcessor()
    queries = processor.process_stream()
    print("Spark streaming queries started. Waiting for termination...")
    processor.spark.streams.awaitAnyTermination()