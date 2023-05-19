# create some plotly visuals to embed within the html
import numpy as np
from top2vec import Top2Vec
import umap.umap_ as umap
import plotly.express as px

model = Top2Vec.load("top2vec_model")

# prepare the data to be visualised
# this needs to include all the topic vectors
# and also the sentence vectors
# and also the word vectors
all_vectors = np.vstack(model.topic_vectors,
                        model.model.docvecs,
                        model.model.wv.vectors)
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(all_vectors)

# in the first visualisation we display only topic vectors

# in the second visualisation we display a single topic vector
# the sentence vectors for that topic
# the word vectors for that topic

df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
fig.write_html("testing_visuals.html")