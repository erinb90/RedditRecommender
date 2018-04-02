from __future__ import division
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, array_contains
from pyspark import SparkConf, SparkContext, SQLContext
import logging
from pyspark.context import SparkContext
def main():


    spark = SparkSession \
        .builder \
        .getOrCreate()
    #--executor-memory 12G
    #--driver-memory 12G
    #--num-executors 5
    spark.sparkContext.setCheckpointDir('gs://reddit_data_soen498/checkpoint/')
    # spark.sparkContext.setCheckpointDir('checkpoint/')
    
    @udf("boolean")
    def isNotDefault(x):
        defaultSubs = ["Art", "AskReddit", "DIY", "Documentaries", "EarthPorn", "Futurology", "GetMotivated", "IAmA", "InternetIsBeautiful", "Jokes", "LifeProTips", "Music", "OldSchoolCool", "Showerthoughts", "UpliftingNews", "announcements", "askscience", "aww", "blog", "books", "creepy", "dataisbeautiful", "explainlikeimfive", "food", "funny", "gadgets", "gaming", "gifs", "history", "listentothis", "mildlyinteresting", "movies", "news", "nosleep", "nottheonion", "personalfinance", "philosophy", "photoshopbattles", "pics", "science", "space", "sports", "television", "tifu", "todayilearned", "videos", "worldnews"]
        return x not in defaultSubs
    
    data = spark.read.json("gs://reddit_data_soen498/RC_2018-01.json")
    # data = spark.read.json("./sample_data.json")
    keep = [data.author, data.id, data.subreddit]
    data = data.select(*keep)
    data = data.filter(data.author != "[deleted]")
    data = data.filter(isNotDefault(data.subreddit))

    data = data.groupBy(data.author).agg(F.collect_set("subreddit").alias("items"))
    size_ = udf(lambda xs: len(xs), IntegerType())
    data = data.filter(size_(data.items) > 1)
    data = data.select(data.items)
    support = 20/data.count()
    fp = FPGrowth(minSupport=support, minConfidence=0.1)
    fpm = fp.fit(data)
    fpm.associationRules.show(100)


    
    fpm.save("gs://reddit_data_soen498/modelFP_noDefaultSub_20support")
    



main()