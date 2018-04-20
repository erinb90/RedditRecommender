# RedditRecommender

### SOEN 498 Project - Winter 2018

Erin Benderoff 27768478 \
Francis Bouchard 26786812

Description of Files: \
```als-recommender.py``` Script that runs ALS algorithm on the sample dataset, evaluates the model and generates top 10 subreddit recommendations for a subset of users.

```frequentitemsets.py``` Script that was run on Google Cloud cluster. Runs FP-Growth algorithm to obtain frequent itemsets and association rules on the February 2018 dataset.

```fp_growth_merge_and_print_results.py``` Script that takes the output of the FP-Growth script (saved as parquet files), merges them into a single dataframe and writes the frequent itemsets to a file in order of descending itemset size and descending frequency.

```frequentLoad.py``` Script that loads the FP-Growth results and displays the association rules.

```als-output.txt``` Output of ALS user recommendations

```fp-growth-output.txt``` Output of FP-Growth frequent itemsets

```association-rules-output.txt``` Output of FP-Growth association rules

```sample_data.json``` Sample data file used for ALS recommender