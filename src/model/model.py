import os
import yaml
import logging
import pickle

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, NGram, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------- LOGGING ---------------- #
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------- UTILS ---------------- #
def get_root_directory():
    return os.getcwd()


def load_params(params_path):
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f"Loaded params from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading params: {e}")
        raise


def load_data(spark, path):
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        df = df.fillna('')
        logger.debug(f"Loaded data from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# ---------------- FEATURE ENGINEERING ---------------- #
def build_pipeline(max_features, ngram_range):
    tokenizer = Tokenizer(inputCol="clean_comment", outputCol="tokens")

    # Only supports fixed n, so we take max ngram
    ngram = NGram(n=ngram_range[1], inputCol="tokens", outputCol="ngrams")

    hashing_tf = HashingTF(
        inputCol="ngrams",
        outputCol="raw_features",
        numFeatures=max_features
    )

    idf = IDF(inputCol="raw_features", outputCol="features")

    label_indexer = StringIndexer(inputCol="category", outputCol="label")

    classifier = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxDepth=5,
        maxIter=100
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        ngram,
        hashing_tf,
        idf,
        label_indexer,
        classifier
    ])

    return pipeline


# ---------------- MAIN ---------------- #
def main():
    try:
        spark = SparkSession.builder \
            .appName("ModelBuilding") \
            .getOrCreate()

        root_dir = get_root_directory()

        params = load_params(os.path.join(root_dir, 'params.yaml'))

        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        train_data = load_data(
            spark,
            os.path.join(root_dir, 'data/interim/train_processed.csv')
        )

        pipeline = build_pipeline(max_features, ngram_range)

        model = pipeline.fit(train_data)

        logger.debug("Model training completed")

        # Save Spark model
        model.save(os.path.join(root_dir, "lgbm_model_spark"))

        logger.debug("Model saved successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()