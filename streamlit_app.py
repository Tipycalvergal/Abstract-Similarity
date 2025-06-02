# plot.py
import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="Talks Similarity Viewer", layout="wide")

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


df = pd.read_csv("vectorised.csv")
df["SIMPLIFIED_TYPE"] = df["TYPE"].apply(smart_title)


unique_types = df["SIMPLIFIED_TYPE"].unique().tolist()
sorted_type_list = sorted(unique_types, key=type_sort_key)
df["TYPE"] = pd.Categorical(df["SIMPLIFIED_TYPE"], categories=sorted_type_list, ordered=True)
df.sort_values("TYPE", inplace=True)

df_plot = pd.DataFrame({
        "Title": df["TITLE"],
        "TYPE_RAW": df["TYPE"],
        "TYPE": df["SIMPLIFIED_TYPE"],
        "Speaker": df["Speaker"]
    })

st.markdown("<h2 style='text-align: center;'>📊 t-SNE Projection of Talks</h2>", unsafe_allow_html=True)
st.markdown("""
---
### 🧭 How to Use
1. **Hover** over any point to explore talk metadata — including the talk **title**, **type**, and **speaker**.
2. **Double-click a category label** in the legend (on the right side) to isolate that talk type.
   - You can also **single-click** other types to toggle them on/off.
""", unsafe_allow_html=True)

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="TYPE",
    color_discrete_sequence=px.colors.qualitative.Alphabet,
    symbol="TYPE",
    hover_data={
        "x": False,  
        "y": False,  
        "TITLE": True,
        "Speaker": True
    },
    title="Talks Similarity (t-SNE View)",
    labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"}
)

fig.update_layout(
    xaxis=dict(range=[-50, 50], autorange=False),  # <- Replace with your own x-range
    yaxis=dict(range=[-60, 60], autorange=False)   # <- Replace with your own y-range
)

st.plotly_chart(fig, use_container_width=True)
