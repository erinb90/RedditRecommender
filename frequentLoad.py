from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth, FPGrowthModel
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, array_contains

def main():
    spark = SparkSession \
        .builder \
        .appName("RedditRecommender") \
        .getOrCreate()
    


    sameModel = FPGrowthModel.load("FPModelGood")
    rules = sameModel.associationRules
    recommendations = rules.where(array_contains("antecedent", "cats"))

    recommendations.orderBy(recommendations.confidence.desc()).show(50)

    
    
    # data.show(200)
    # model.save("frequent")
    



main()