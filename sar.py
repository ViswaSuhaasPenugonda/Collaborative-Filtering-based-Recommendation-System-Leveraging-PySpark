import sys
import numpy as np
import pandas as pd
import papermill as pm
import itertools
import logging
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
import os

os.chdir('data_science/PythonCode/recommeder-system-for-amazon-products/')
TOP_K=10

df = pd.read_csv('rating.csv',nrows=100000,names=['userId','productId','Rating','timestamp'])


###3.2 Split the data using the python random splitter provided in utilities:¶
df.columns

header = {
    "col_user": "userId",
    "col_item": "productId",
    "col_rating": "Rating",
    "col_timestamp": "timestamp",
    "col_prediction": "Prediction",
}

train, test = python_stratified_split(df, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"])

# set log level to INFO

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SARSingleNode(similarity_type="jaccard", time_decay_coefficient=30, time_now=None, timedecay_formula=True,
    **header)

model.fit(train)

top_k = model.recommend_k_items(test, remove_seen=True)


top_k.head()
##3.3 Evaluate the results
args = [test, top_k]
kwargs = dict(col_user='userId',
              col_item='productId',
              col_rating='Rating',
              col_prediction='Prediction',
              relevancy_method='top_k',
              k=TOP_K)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)



print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')
