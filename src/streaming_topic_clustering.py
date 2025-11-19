"""Topic clustering on streaming news headlines.

This script reuses the Word2Vec and KMeans models trained in train_word2vec_kmeans.py
to assign topic clusters to incoming, streaming news headlines.
It uses Spark Structured Streaming with a directory source.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, split
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.clustering import KMeansModel

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STREAM_INPUT_DIR = "/home/farouk/streaming_input"  # adjust as needed
WORD2VEC_MODEL_PATH = "models/word2vec_news"
KMEANS_MODEL_PATH = "models/kmeans_news"


def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Streaming-Topic-Clustering")
        .getOrCreate()
    )


def main():
    spark = create_spark()

    # Load trained models
    w2v_model = Word2VecModel.load(WORD2VEC_MODEL_PATH)
    kmeans_model = KMeansModel.load(KMEANS_MODEL_PATH)

    # Create streaming DataFrame from a directory of text files
    lines = spark.readStream.text(STREAM_INPUT_DIR)

    # Basic preprocessing
    words_df = lines.withColumn(
        "tokens",
        split(lower(col("value")), " ")
    )

    # Vectorise using the trained Word2Vec model
    vectorised = w2v_model.transform(words_df)

    # Predict cluster for each incoming line
    predictions = kmeans_model.transform(vectorised)

    # For demo purposes, just show the raw line and its predicted cluster id
    query = (
        predictions
        .select(col("value").alias("headline"), col("prediction").alias("cluster_id"))
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
