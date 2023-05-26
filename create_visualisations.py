# create some plotly visuals to embed within the html
import numpy as np
from top2vec import Top2Vec
from umap import UMAP
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import plotly
import pickle

model = Top2Vec.load("top2vec_model")

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

# in the first visualisation we display only topic vectors
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
                                       width=2
                                       )
                         ),
                         text=[', '.join(wordvec[:5]) for wordvec in model.topic_words],
                         hoverinfo='text',
                         ))

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.write_html("testing_visuals.html", full_html=False, include_plotlyjs='cdn')

# in the second visualisation we display a single topic vector
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
fig.write_html("testing_visuals_second_plot.html", full_html=False, include_plotlyjs='cdn')
