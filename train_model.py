# this file is used for eda on raw text
import pickle
import numpy as np
from top2vec import Top2Vec

# get data
filename = 'sentences_from_master_dict_2023-05-05.pkl'
with open(filename, 'rb') as f:
    sentences_object = pickle.load(f)

# train and save model
model = Top2Vec(sentences_object.sentences)
model.save("top2vec_model")

# investigate model
all_topic_sizes, all_topic_nums = model.get_topic_sizes()

keywords = ["marine", "coastal", "seas", "oceans", "restore"]
marine_topic_nums = []
for kw in keywords:
    _, _, _, topic_nums = model.search_topics(keywords=[kw], num_topics=5)
    marine_topic_nums.extend(topic_nums)
marine_topic_nums = list(set(marine_topic_nums))
marine_topic_sizes = np.asarray(all_topic_sizes)[marine_topic_nums]
marine_topic_words = model.topic_words[marine_topic_nums, :]

words, word_scores = model.similar_words(keywords=["ocean"], keywords_neg=[], num_words=20)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")