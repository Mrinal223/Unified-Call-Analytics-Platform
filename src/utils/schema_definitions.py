# src/utils/schema_definitions.py
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, TimestampType

# Schema for the raw JSON messages coming from Kafka
raw_call_schema = StructType([
    StructField("call_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("filename", StringType(), True),
    StructField("transcript", StringType(), True),
    StructField("duration_seconds", IntegerType(), True),
    StructField("sentiment", StringType(), True),
    StructField("polarity_score", DoubleType(), True),
    StructField("problem_resolved", BooleanType(), True),
    StructField("issue_description", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("call_summary_one_line", StringType(), True),
    StructField("agent_sentiment", StringType(), True),
    StructField("agent_sentiment_score", DoubleType(), True),
    StructField("agent_experience_level", StringType(), True),
    StructField("issue_complexity", StringType(), True),
    StructField("resolved_chance", DoubleType(), True),
    StructField("resolution_status", StringType(), True),
    StructField("agent_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("call_type", StringType(), True)
])

# Schema for the processed data written to Parquet (adds processing_time)
processed_call_schema = StructType([
    StructField("call_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("processing_time", TimestampType(), True),
    StructField("filename", StringType(), True),
    StructField("transcript", StringType(), True),
    StructField("duration_seconds", IntegerType(), True),
    StructField("sentiment", StringType(), True),
    StructField("polarity_score", DoubleType(), True),
    StructField("problem_resolved", BooleanType(), True),
    StructField("issue_description", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("call_summary_one_line", StringType(), True),
    StructField("agent_sentiment", StringType(), True),
    StructField("agent_sentiment_score", DoubleType(), True),
    StructField("agent_experience_level", StringType(), True),
    StructField("issue_complexity", StringType(), True),
    StructField("resolved_chance", DoubleType(), True),
    StructField("resolution_status", StringType(), True),
    StructField("agent_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("call_type", StringType(), True),
    StructField("hour_of_day", IntegerType(), True),
    StructField("transcript_length", IntegerType(), True)
])