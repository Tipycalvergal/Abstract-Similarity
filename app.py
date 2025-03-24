import os
import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import spacy
import plotly.express as px
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# 🔧 Functions
def smart_title(text):
    text = str(text).strip()
    if text.upper().startswith("MS"):
        prefix, rest = text.split(":", 1)
        return prefix.upper() + ":" + rest.strip().title()
    else:
        return text.title()

def extract_ms_code(text):
    text = str(text)
    match = re.match(r"(MS\d+)", text.upper())
    return match.group(1) if match else text

def type_sort_key(t):
    t = str(t)
    if t == "Plenary Talk":
        return (0, "")
    elif t == "Contributed Talk":
        return (1, "")
    else:
        code = extract_ms_code(t)
        num = int(re.findall(r"\d+", code)[0]) if re.findall(r"\d+", code) else 999
        return (2, num)

# 📄 Load data
df = pd.read_csv('registration-roster - talks.csv', usecols=["TITLE", "ABSTRACT", "TYPE", "FIRST_NAME", "LAST_NAME"])
df.fillna("", inplace=True)
df["COMBINED"] = df["TITLE"] + " " + df["ABSTRACT"]
df["FULL_NAME"] = df["FIRST_NAME"] + " " + df["LAST_NAME"]
df["TYPE_RAW"] = df["TYPE"]
df["SIMPLIFIED_TYPE"] = df["TYPE"].apply(smart_title)

# 🌐 Load embeddings
nlp = spacy.load('en_core_web_lg')
combined_vectors = list(nlp.pipe(df["COMBINED"].tolist()))
vector_list = np.array([doc.vector for doc in combined_vectors])

# 📊 Similarity + TSNE
similarity_matrix = cosine_similarity(vector_list)
tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate="auto")
tsne_results = tsne.fit_transform(vector_list)

# 📌 Build plot DataFrame
df_plot = pd.DataFrame({
    "x": tsne_results[:, 0],
    "y": tsne_results[:, 1],
    "TITLE": df["TITLE"],
    "TYPE_RAW": df["TYPE_RAW"],
    "TYPE": df["SIMPLIFIED_TYPE"],
    "Speaker": df["FULL_NAME"]
})

# ✅ Order by TYPE
df_order = pd.read_csv('MS_Topics_Combined_String.csv', usecols=["MS Topic"])
df_order["MS Topic"] = df_order["MS Topic"].apply(smart_title)

unique_types = df_plot["TYPE"].unique().tolist()
sorted_type_list = sorted(unique_types, key=type_sort_key)
df_plot["TYPE"] = pd.Categorical(df_plot["TYPE"], categories=sorted_type_list, ordered=True)
df_plot.sort_values("TYPE", inplace=True)

# 🎨 Create figure
fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="TYPE",
    symbol="TYPE",
    hover_data={
        "TITLE": True,
        "TYPE_RAW": True,
        "Speaker": True,
        "x": False,
        "y": False
    },
    title="Talks Similarity (t-SNE View)",
    labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"}
)

# 🌐 Dash App
app = dash.Dash(__name__)
app.title = "Talks Similarity Explorer"

app.layout = html.Div([
    html.H1("Talks Similarity Dashboard", style={"textAlign": "center"}),
    html.P("Visualizing similarity among talks using t-SNE and spaCy embeddings.", style={"textAlign": "center"}),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port)
