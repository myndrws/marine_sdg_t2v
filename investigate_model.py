# investigate the trained model
import numpy as np
import pandas as pd
from top2vec import Top2Vec
import pickle
from pprint import pprint

### open things
model = Top2Vec.load("top2vec_model")
with open('sentences_from_master_dict_2023-05-05.pkl', 'rb') as f:
    sentences_object = pickle.load(f)
with open('master_dict_2023-05-05.pkl', 'rb') as f:
    master_dict = pickle.load(f)

# investigate model
all_topic_sizes, all_topic_nums = model.get_topic_sizes()

keywords = ["marine", "coastal", "seas", "oceans", "restoration", "health"]
m_topic_words, m_word_scores, m_topic_scores, m_topic_nums = model.search_topics(keywords=keywords,
                                                                                 num_topics=5)
words, word_scores = model.similar_words(keywords=["ocean"], keywords_neg=[], num_words=20)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")

# create a big dataframe with everything in it
marine_docs = {}
for topic in m_topic_nums:
    topic_size = model.topic_sizes.loc[topic]
    mdocs, mscores, mids = model.search_documents_by_topic(topic_num=topic, num_docs=topic_size)
    marine_docs[f"topic_{topic}"] = pd.DataFrame({"Documents": mdocs,
                                                  "Scores": mscores,
                                                  "IDs": mids})
marine_docs_df = pd.concat(marine_docs)
marine_docs_df["uids"] = np.asarray(sentences_object.uids)[list(marine_docs_df.IDs)]
uids_split = [uids.split("_") for uids in list(marine_docs_df.uids)]
marine_docs_df["doc_title"] = [master_dict[uid[0] + "_" + uid[1]]["title"] for uid in uids_split]
marine_docs_df["doc_date"] = [master_dict[uid[0] + "_" + uid[1]]["updated"] for uid in uids_split]
marine_docs_df["doc_url"] = [master_dict[uid[0] + "_" + uid[1]]["url"] for uid in uids_split]
unique_docs = set(marine_docs_df.doc_title)


def makeParaFromSentence(all_sentences_object, sentence_uid, n_sentence_either_side):
    page_id, doc_id, sentence_id = sentence_uid.split("_")
    para = ""
    for ind, uid in enumerate(all_sentences_object.uids):
        if uid.startswith(f"{page_id}_{doc_id}_"):
            this_doc_id = int(uid.split("_")[2])
            cond1 = this_doc_id >= (int(sentence_id) - n_sentence_either_side)
            cond2 = this_doc_id <= (int(sentence_id) + n_sentence_either_side)
            if cond1 and cond2:
                para += " " + all_sentences_object.sentences[ind]
    return para

example_para = makeParaFromSentence(all_sentences_object=sentences_object,
                                    sentence_uid="29_17_870",
                                    n_sentence_either_side=3)

pprint(example_para)