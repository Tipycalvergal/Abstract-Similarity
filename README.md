# 📚 Abstract Similarity Dashboard

An interactive Streamlit application that visualizes the semantic similarity between academic talk titles and abstracts using spaCy embeddings and t-SNE projection.

## 🔍 Overview

This dashboard allows users to explore how closely related different academic talks are based on their titles and abstracts. By leveraging natural language processing techniques, the application projects talks into a 2D space where proximity indicates semantic similarity.

## ✨ Features

- **Semantic Embedding**: Utilizes `spaCy`'s `en_core_web_lg` model to convert text into semantic vectors.
- **Dimensionality Reduction**: Applies t-SNE to reduce high-dimensional embeddings into 2D for visualization.
- **Interactive Visualization**: Employs Plotly to create an interactive scatter plot of talks.
- **User-Controlled Vectorization**: Provides a button within the app to recompute embeddings as needed.
- **Categorization**: Colors and symbols represent different talk types for easy identification.

