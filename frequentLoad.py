from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowthModel

def main():
    spark = SparkSession \
        .builder \
        .appName("RedditRecommender") \
        .getOrCreate()

    sameModel = FPGrowthModel.load("gs://redditrecommender/fp-growth-results")
    rules = sameModel.associationRules

    rules.orderBy(rules.confidence.desc()).show(100, truncate=False)


main()