# this file is used for eda on raw text
import pickle
import numpy as np
from top2vec import Top2Vec
from extract_data import etl_data

# get data
filename = 'sentences_from_master_dict_2023-05-05.pkl'
with open(filename, 'rb') as f:
    sentences_object = pickle.load(f)

# train and save model
umap_args = {'n_neighbors': 15,
             'n_components': 5,
             'metric': 'cosine'}
hdbscan_args = {'min_cluster_size': 50,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'}

model = Top2Vec(sentences_object.sentences,
                min_count=20,
                topic_merge_delta=0.2,
                umap_args=umap_args,
                hdbscan_args=hdbscan_args)
model.save("top2vec_model")

