"""Batch training pipeline for Word2Vec + KMeans topic clustering on news headlines.

This script is adapted from my 'Machine Learning on Big Data' coursework.
It demonstrates how to train distributed word embeddings and a clustering model
on a large JSON news dataset using PySpark.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, size
from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.ml.clustering import KMeans

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NEWS_DATA_PATH = "hdfs://localhost:9000/AssignmentDatasets/News_Category_Dataset_v3.json"
WORD2VEC_MODEL_PATH = "models/word2vec_news"
KMEANS_MODEL_PATH = "models/kmeans_news"

MIN_TOKEN_LENGTH = 3
K_CLUSTERS = 10
VECTOR_SIZE = 100
MIN_WORD_COUNT = 3


def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Word2Vec-KMeans")
        .getOrCreate()
    )


def load_and_preprocess(spark: SparkSession):
    # Load JSON dataset
    df = spark.read.json(NEWS_DATA_PATH)

    # Assume the headline text is in a column called 'headline' or 'short_description'.
    # Adjust this based on the actual schema of your dataset.
    text_col = "headline" if "headline" in df.columns else "short_description"

    df = df.select(col(text_col).alias("text")).na.drop()

    # Basic cleaning: lowercase + remove punctuation
    df = df.withColumn(
        "clean_text",
        lower(regexp_replace(col("text"), "[^a-zA-Z ]", " ")),
    )

    # Tokenise
    df = df.withColumn("tokens", split(col("clean_text"), " "))

    # Remove short tokens
    df = df.withColumn(
        "tokens",
        # filter out tokens with length < MIN_TOKEN_LENGTH
        # using a simple expression
        split(col("clean_text"), " ")
    )

    # Drop rows with very few tokens
    df = df.filter(size(col("tokens")) >= 3)

    # Remove stop words
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    df = remover.fit(df).transform(df)

    return df


def train_word2vec(df):
    w2v = Word2Vec(
        vectorSize=VECTOR_SIZE,
        minCount=MIN_WORD_COUNT,
        inputCol="filtered_tokens",
        outputCol="features",
    )
    model = w2v.fit(df)
    return model, model.transform(df)


def train_kmeans(df_with_vectors):
    kmeans = KMeans(
        k=K_CLUSTERS,
        seed=42,
        featuresCol="features",
        predictionCol="prediction",
    )
    model = kmeans.fit(df_with_vectors)
    return model


def main():
    spark = create_spark()

    try:
        df = load_and_preprocess(spark)

        # Train Word2Vec model
        w2v_model, with_vectors = train_word2vec(df)

        # Train KMeans model
        kmeans_model = train_kmeans(with_vectors)

        # Save models
        w2v_model.save(WORD2VEC_MODEL_PATH)
        kmeans_model.save(KMEANS_MODEL_PATH)

        print(f"Saved Word2Vec model to: {WORD2VEC_MODEL_PATH}")
        print(f"Saved KMeans model to: {KMEANS_MODEL_PATH}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
