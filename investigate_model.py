# investigate the trained model
import numpy as np
from top2vec import Top2Vec

model = Top2Vec.load("top2vec_model")

# investigate model
all_topic_sizes, all_topic_nums = model.get_topic_sizes()

keywords = ["marine", "coastal", "seas", "oceans"]
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