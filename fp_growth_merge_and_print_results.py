from pyspark.sql import SparkSession

def main():
    spark = SparkSession \
        .builder \
        .getOrCreate()

    output_file = open('fp-growth-output.txt', 'w')

    mergedDF = spark.read.option("mergeSchema", "true").parquet("./data")

    for row in mergedDF.collect():
        output_file.write('Items: \n')
        items = []
        for item in row['items']:
            items.append(item)
        output_file.write('{' + ','.join(items) + '}\n')
        output_file.write('Freq: ' + str(row['freq']) + '\n\n')


main()
