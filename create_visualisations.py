# create some plotly visuals to embed within the html
import numpy as np
from top2vec import Top2Vec
from umap import UMAP
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
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

# in the first visualisation we display only topic vectors
# the other points (sentences and words) are greyed out dots,
# and we can't hover over them
colours = embeddings[(all_vector_levels == 0).squeeze(), 0] * embeddings[(all_vector_levels == 0).squeeze(), 1]
sizes = (model.topic_sizes-min(model.topic_sizes))/(max(model.topic_sizes)-min(model.topic_sizes)) * 100
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig = go.Figure(layout=layout)

# Add trace
fig.add_trace(go.Scatter(x=embeddings[(all_vector_levels == 0).squeeze(), 0],
                         y=embeddings[(all_vector_levels == 0).squeeze(), 1],
                         mode='markers',
                         name='topics',
                         marker=dict(
                             size=sizes,
                             color=colours,
                             colorscale='plasma'
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
fig.add_trace(go.Scatter(x=embeddings[(all_vector_levels != 0).squeeze(), 0],
                         y=embeddings[(all_vector_levels != 0).squeeze(), 1],
                         mode='markers',
                         name='sentences and words',
                         marker=dict(
                             size=2,
                             color='grey',  # set color equal to a variable
                             opacity=0.5
                         ),
                         hoverinfo="skip"
                         ))