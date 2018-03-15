from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import hash

def main():
    spark = SparkSession \
        .builder \
        .appName("RedditRecommender") \
        .getOrCreate()

    data = spark.read.json("./sample_data.json")

    keep = [data.author, data.id, data.subreddit]
    data = data.select(*keep)

    data = data.withColumn('author_id', hash(data.author))
    data = data.withColumn('subreddit_id', hash(data.subreddit))
    # data.show(50)

    subreddit_count = data.groupBy([data.author_id, data.subreddit_id]).count().orderBy(data.author_id)

    #subreddit_count.show(50)
    #
    # # parts = lines.map(lambda row: row.value.split("::"))
    # # ratingsRDD = parts.map(lambda p: Row(userId=int(p.author), itemId=int(p[1]),
    # #                                      rating=float(p[2]), timestamp=int(p[3])))
    ratings = subreddit_count
    (training, test) = ratings.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="author_id", itemCol="subreddit_id", ratingCol="count",
              coldStartStrategy="drop", implicitPrefs=True)
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(str(rmse))
    predictions.show(50)

    userRecs = model.recommendForAllUsers(10)
    userRecs.show()


main()