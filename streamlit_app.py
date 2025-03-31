import streamlit as st
import pandas as pd
import numpy as np
import spacy
import spacy.cli
import plotly.express as px
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

st.set_page_config(page_title="Talks Similarity Viewer", layout="wide")

try:
    nlp = spacy.load('en_core_web_lg')
except OSError:
    with st.spinner("Downloading spaCy model..."):
        spacy.cli.download("en_core_web_lg")
    nlp = spacy.load('en_core_web_lg')

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

st.title("📚 Talks Similarity Dashboard")
st.markdown("""

### 🔍 What Does This App Do?

This dashboard analyzes talk **titles** and **abstracts**, projects them into a 2D space using t-SNE, and visualizes how similar they are based on their semantic content.

Each point in the scatter plot represents a talk. Talks that are **closer together** are more **semantically similar**, while those farther apart are more different in content.

---

### 🧠 How It Works

- 🧬 **Text Embedding**: We use [`spaCy`](https://spacy.io/) to convert text into semantic vectors.
- 📉 **t-SNE Projection**: These vectors are reduced to two dimensions using t-SNE for visualization.
- 📊 **Interactive Plot**: Talks are colored by their type (e.g., *Plenary Talk*, *MS01*, etc.) for easy identification.

""", unsafe_allow_html=True)

# 📄 Load data
with st.spinner("Loading data..."):
    df = pd.read_csv('talks.csv', usecols=["TITLE", "ABSTRACT", "TYPE", "FIRST_NAME", "LAST_NAME"])
    df.fillna("", inplace=True)
    df["COMBINED"] = df["TITLE"] + " " + df["ABSTRACT"]
    df["FULL_NAME"] = df["FIRST_NAME"] + " " + df["LAST_NAME"]
    df["TYPE_RAW"] = df["TYPE"]
    df["SIMPLIFIED_TYPE"] = df["TYPE"].apply(smart_title)

    combined_vectors = list(nlp.pipe(df["COMBINED"].tolist()))
    vector_list = np.array([doc.vector for doc in combined_vectors])

    similarity_matrix = cosine_similarity(vector_list)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate="auto")
    tsne_results = tsne.fit_transform(vector_list)

    df_plot = pd.DataFrame({
        "x": tsne_results[:, 0],
        "y": tsne_results[:, 1],
        "TITLE": df["TITLE"],
        "TYPE_RAW": df["TYPE_RAW"],
        "TYPE": df["SIMPLIFIED_TYPE"],
        "Speaker": df["FULL_NAME"]
    })

    df_order = pd.read_csv('MS_Topics_Combined_String.csv', usecols=["MS Topic"])
    df_order["MS Topic"] = df_order["MS Topic"].apply(smart_title)

    unique_types = df_plot["TYPE"].unique().tolist()
    sorted_type_list = sorted(unique_types, key=type_sort_key)
    df_plot["TYPE"] = pd.Categorical(df_plot["TYPE"], categories=sorted_type_list, ordered=True)
    df_plot.sort_values("TYPE", inplace=True)


st.subheader("📊 t-SNE Projection of Talks")
st.markdown("""
### 🧭 How to Use
1. **Hover** over any point to explore talk metadata — including the talk **title**, **type**, and **speaker**.
2. **Double-click a category label** in the legend (on the right side) to isolate that talk type.
   - You can also **single-click** other types to toggle them on/off.
---
""", unsafe_allow_html=True)

fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="TYPE",
    color_discrete_sequence=px.colors.qualitative.Alphabet,
    symbol="TYPE",
    hover_data={
        "x": False,  
        "y": False,  
        "TITLE": True,
        "TYPE_RAW": False,
        "Speaker": True
    },
    title="Talks Similarity (t-SNE View)",
    labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"}
)

st.plotly_chart(fig, use_container_width=True)
