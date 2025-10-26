# src/services/aggregation/spark_aggregator_to_postgres.py
import os
import sys
sys.path.append('/app')
sys.path.append('/app/src')
import psycopg2
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, count, avg, sum, when, col, to_timestamp, lit, coalesce
from pyspark.sql.types import TimestampType, DoubleType
from dotenv import load_dotenv
from src.utils.schema_definitions import processed_call_schema
from datetime import datetime

# ADDED: Add project root to sys.path for local module imports
sys.path.append('/app')

load_dotenv()

PARQUET_PROCESSED_PATH = os.getenv("PARQUET_PROCESSED_PATH", "/data/data_lake/processed_calls_parquet")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgresql")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "call_analytics_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
CHECKPOINT_BASE_PATH = os.getenv("CHECKPOINT_BASE_PATH", "/tmp/spark_checkpoints")

JDBC_DRIVER = "org.postgresql.Driver"
JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

class SparkAggregatorToPostgres:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("SparkAggregatorToPostgres") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.5.0") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        print("SparkSession created for aggregation to PostgreSQL.")

    def aggregate_and_write(self):
        df = self.spark.readStream \
            .format("parquet") \
            .schema(processed_call_schema) \
            .load(PARQUET_PROCESSED_PATH)

        WATERMARK_INTERVAL = "1 hour"
        WINDOW_DURATION = "1 hour"
        SLIDE_DURATION = "1 hour"

        agent_performance_hourly = df \
            .withWatermark("timestamp", WATERMARK_INTERVAL) \
            .groupBy(
                window(col("timestamp"), WINDOW_DURATION, SLIDE_DURATION).alias("window"),
                "agent_id"
            ) \
            .agg(
                count("*").alias("total_calls"),
                sum(when(col("problem_resolved") == True, 1).otherwise(0)).alias("resolved_calls"),
                avg("duration_seconds").alias("avg_call_duration"),
                avg("polarity_score").alias("avg_sentiment_score")
            ) \
            .withColumn("resolution_rate", (col("resolved_calls").cast(DoubleType()) / col("total_calls")).cast(DoubleType())) \
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                "agent_id",
                "total_calls",
                "resolved_calls",
                coalesce(col("resolution_rate"), lit(0.0)).alias("resolution_rate"),
                coalesce(col("avg_call_duration"), lit(0.0)).alias("avg_call_duration"),
                coalesce(col("avg_sentiment_score"), lit(0.0)).alias("avg_sentiment_score"),
                lit(datetime.now()).cast(TimestampType()).alias("last_updated")
            )

        product_issue_trends_hourly = df \
            .withWatermark("timestamp", WATERMARK_INTERVAL) \
            .groupBy(
                window(col("timestamp"), WINDOW_DURATION, SLIDE_DURATION).alias("window"),
                "product_category",
                "issue_description"
            ) \
            .agg(
                count("*").alias("call_count"),
                avg("polarity_score").alias("avg_sentiment_category")
            ) \
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                "product_category",
                "issue_description",
                "call_count",
                coalesce(col("avg_sentiment_category"), lit(0.0)).alias("avg_sentiment_category"),
                lit(datetime.now()).cast(TimestampType()).alias("last_updated")
            )

        def upsert_to_postgres(df, epoch_id, table_name, primary_keys):
            if df.isEmpty(): return
            pandas_df = df.toPandas()
            conn = None
            try:
                conn = psycopg2.connect(host=POSTGRES_HOST, database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, port=POSTGRES_PORT)
                cursor = conn.cursor()
                columns = pandas_df.columns.tolist()
                columns_str = ", ".join(columns)
                placeholders = ", ".join([f"%({col})s" for col in columns])
                update_set_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col not in primary_keys])
                if not primary_keys: raise ValueError("Primary keys must be provided.")
                conflict_target = ", ".join(primary_keys)
                upsert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set_str};"
                for index, row in pandas_df.iterrows():
                    cursor.execute(upsert_sql, row.to_dict())
                conn.commit()
                print(f"  Batch {epoch_id} successfully upserted {len(pandas_df)} rows to {table_name}.")
            except Exception as e:
                print(f"  ❌ Error during PostgreSQL upsert for batch {epoch_id} to {table_name}: {e}")
                if conn: conn.rollback()
            finally:
                if conn: conn.close()

        query_agent_perf = agent_performance_hourly \
            .writeStream \
            .foreachBatch(lambda df, epoch_id: upsert_to_postgres(df, epoch_id, "agent_performance_hourly", ["window_start", "agent_id"])) \
            .outputMode("update") \
            .option("checkpointLocation", os.path.join(CHECKPOINT_BASE_PATH, "agg_agent_postgres")) \
            .trigger(processingTime='60 seconds') \
            .start()
        print("Streaming hourly agent performance to PostgreSQL.")

        query_product_trends = product_issue_trends_hourly \
            .writeStream \
            .foreachBatch(lambda df, epoch_id: upsert_to_postgres(df, epoch_id, "product_issue_trends_hourly", ["window_start", "product_category", "issue_description"])) \
            .outputMode("update") \
            .option("checkpointLocation", os.path.join(CHECKPOINT_BASE_PATH, "agg_product_postgres")) \
            .trigger(processingTime='60 seconds') \
            .start()
        print("Streaming hourly product/issue trends to PostgreSQL.")

        return query_agent_perf, query_product_trends

    def get_parquet_schema(self):
        from src.utils.schema_definitions import processed_call_schema
        return processed_call_schema

if __name__ == "__main__":
    aggregator = SparkAggregatorToPostgres()
    queries = aggregator.aggregate_and_write()
    print("Spark aggregation queries started. Waiting for termination...")
    aggregator.spark.streams.awaitAnyTermination()