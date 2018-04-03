from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import hash, udf


def main():
    spark = SparkSession \
        .builder \
        .appName("RedditRecommender") \
        .getOrCreate()

    data = spark.read.json("./sample_data.json")

    cols_to_keep = [data.author, data.id, data.subreddit]
    data = data.select(*cols_to_keep)
    data = data.filter(data.author != "[deleted]")

    @udf("boolean")
    def isNotDefault(x):
        defaultSubs = ["Art", "AskReddit", "DIY", "Documentaries", "EarthPorn", "Futurology", "GetMotivated", "IAmA",
                       "InternetIsBeautiful", "Jokes", "LifeProTips", "Music", "OldSchoolCool", "Showerthoughts",
                       "UpliftingNews", "announcements", "askscience", "aww", "blog", "books", "creepy",
                       "dataisbeautiful", "explainlikeimfive", "food", "funny", "gadgets", "gaming", "gifs", "history",
                       "listentothis", "mildlyinteresting", "movies", "news", "nosleep", "nottheonion",
                       "personalfinance", "philosophy", "photoshopbattles", "pics", "science", "space", "sports",
                       "television", "tifu", "todayilearned", "videos", "worldnews"]
        return x not in defaultSubs

    data = data.filter(isNotDefault(data.subreddit))

    data = data.groupBy([data.author, data.subreddit]).count().orderBy(data.author)
    data = data.withColumn('author_id', hash(data.author))
    data = data.withColumn('subreddit_id', hash(data.subreddit))

    (training, test) = data.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="author_id", itemCol="subreddit_id", ratingCol="count",
              coldStartStrategy="drop", implicitPrefs=True)
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print('Root mean squared error: ' + str(rmse))

    users = data.select(als.getUserCol()).distinct().limit(3)
    user_subset_recs = model.recommendForUserSubset(users, 5)

    subreddit_recs = {}

    for row in user_subset_recs.collect():
        author = get_author_from_id(data, row['author_id'])
        subreddit_recs[author] = []
        for rec in row['recommendations']:
            subreddit_recs[author].append(get_subreddit_from_id(data, rec[0]))

    for author in subreddit_recs.keys():
        print('Recommendations for user ' + author)
        for rec in subreddit_recs[author]:
            print(rec)
        print()


def get_subreddit_from_id(df, subreddit_id):
    df_copy = df.filter(df.subreddit_id == subreddit_id)
    if df_copy.select(df.subreddit).collect():
        subreddit = df_copy.select(df.subreddit).collect()[0][0]
    else:
        subreddit = 'null'
    return subreddit


def get_author_from_id(df, author_id):
    df_copy = df.filter(df.author_id == author_id)
    if df_copy.select(df.author).collect():
        author = df_copy.select(df.author).collect()[0][0]
    else:
        author = 'null'
    return author


main()
