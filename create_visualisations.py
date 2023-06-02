# create some plotly visuals to embed within the html
import numpy as np
import pandas as pd
from top2vec import Top2Vec
from umap import UMAP
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import plotly
import pickle

model = Top2Vec.load("top2vec_model")
marine_topics = [407, 144, 159, 163, 351]

# prepare the data to be visualised
# this needs to include all the topic vectors
# and also the sentence vectors
# and also the word vectors
all_vectors = np.vstack([model.topic_vectors,
                         model.model.dv.vectors,
                         model.model.wv.vectors])
all_vectors = normalize(all_vectors)
all_vector_levels = np.concatenate([np.repeat(0, len(model.topic_vectors)),
                                    np.repeat(1, len(model.model.dv.vectors)),
                                    np.repeat(2, len(model.model.wv.vectors))]).reshape(-1, 1)
reducer = UMAP(random_state=42)
embeddings = reducer.fit_transform(all_vectors)
with open('2d_umap_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# topics only umap
topics_only_embeds = reducer.fit_transform(model.topic_vectors)

# separate into two data arrays for two different traces
marine_topic_embeds = topics_only_embeds[marine_topics, :]
not_marine_mask = np.setdiff1d(list(range(len(topics_only_embeds))), marine_topics)
rest_topic_embeds = topics_only_embeds[not_marine_mask, :]

########################################################

# in the first visualisation we display only topic vectors
# and we display ALL of them, brightly coloured
colours = topics_only_embeds[:, 0] * topics_only_embeds[:, 1]
sizes = (model.topic_sizes - min(model.topic_sizes)) / (max(model.topic_sizes) - min(model.topic_sizes)) * 100
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=topics_only_embeds[:, 0],
                         y=topics_only_embeds[:, 1],
                         mode='markers',
                         name='topics',
                         marker=dict(
                             size=sizes,
                             sizemin=5,
                             color=colours,
                             colorscale='plasma',
                             line=dict(color='black',
                                       width=1
                                       )
                         ),
                         text=[', '.join(wordvec[:5]) for wordvec in model.topic_words],
                         hoverinfo='text',
                         ))

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.write_html("visual_one.html", full_html=False, include_plotlyjs='cdn')

########################################################################

# in the second visual we turn the key topics blue, grey out the rest, and zoom in on animation
sizes = (model.topic_sizes - min(model.topic_sizes)) / (max(model.topic_sizes) - min(model.topic_sizes)) * 100
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=rest_topic_embeds[:, 0],
                         y=rest_topic_embeds[:, 1],
                         mode='markers',
                         name='topics',
                         marker=dict(
                             size=sizes[not_marine_mask],
                             sizemin=5,
                             color="grey",
                             opacity=0.3,
                             line=dict(color='grey',
                                       width=1
                                       )
                         ),
                         text=[', '.join(wordvec) for wordvec in np.asarray(model.topic_words)[not_marine_mask, :5]],
                         hoverinfo='text',
                         ))
fig.add_trace(go.Scatter(x=marine_topic_embeds[:, 0],
                         y=marine_topic_embeds[:, 1],
                         mode='markers',
                         name='topics',
                         marker=dict(
                             size=sizes[marine_topics],
                             sizemin=5,
                             opacity=1,
                             color="#57edf7",
                             line=dict(color='black',
                                       width=2
                                       )
                         ),
                         text=[', '.join(wordvec[:5]) for wordvec in np.asarray(model.topic_words)[marine_topics, :5]],
                         hoverinfo='text',
                         ))
fig.update_layout(showlegend=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.write_html("visual_two.html", full_html=False, include_plotlyjs='cdn')


# in the third visualisation we display a single topic vector
# the sentence vectors for that topic
# the word vectors for that topic
# try this for one topic now, might build dynamic visual after
chosen_topic = 159
# get all the documents that are that topic
chosen_topic_docs_filter = np.asarray(model.document_ids[model.doc_top == chosen_topic])
chosen_topic_docs_vector_inds = chosen_topic_docs_filter + len(model.topic_vectors)
chosen_topic_docs = list(np.asarray(model.documents)[model.doc_top == chosen_topic])
# then get the 50 words closest to topic center to also visualise
chosen_topic_words = np.asarray(model.topic_words)[chosen_topic, :]
chosen_topic_words_filter = np.asarray([model.model.wv.index_to_key.index(word) for word in chosen_topic_words])
chosen_topic_words_vector_inds = chosen_topic_words_filter + len(model.topic_vectors) + len(model.model.dv.vectors)

layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig = go.Figure(layout=layout)

fig.add_trace(go.Scatter(x=embeddings[chosen_topic_docs_vector_inds, 0],
                         y=embeddings[chosen_topic_docs_vector_inds, 1],
                         mode='markers',
                         name='Sentences',
                         marker=dict(
                             size=10,
                             color='#9ec0f7',
                             line=dict(color='black',
                                       width=0.5
                                       )
                         ),
                         text=chosen_topic_docs,
                         hoverinfo="text"
                         ))
fig.add_trace(go.Scatter(x=embeddings[chosen_topic_words_vector_inds, 0],
                         y=embeddings[chosen_topic_words_vector_inds, 1],
                         mode='markers',
                         name='Words',
                         marker=dict(
                             size=10,
                             color='#f79ec9',
                             line=dict(color='black',
                                       width=0.5
                                       )
                         ),
                         text=chosen_topic_words,
                         hoverinfo="text"
                         ))
fig.add_trace(go.Scatter(x=np.asarray(embeddings[chosen_topic, 0]),
                         y=np.asarray(embeddings[chosen_topic, 1]),
                         mode='markers',
                         name='Topic',
                         marker=dict(
                             symbol="diamond",
                             size=20,
                             color='#facf6b',
                             line=dict(color='black',
                                       width=2
                                       )
                         ),
                         text=', '.join(model.topic_words[chosen_topic, :10]),
                         hoverinfo="text"
                         ))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.write_html("visual_3.html", full_html=False, include_plotlyjs='cdn')


#####################################################
