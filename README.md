# Talks Similarity Dashboard

This is a Dash web application that visualizes the similarity between conference talks using spaCy embeddings and t-SNE.

## Features

- t-SNE plot of talk embeddings
- Interactive tooltips with title, type, and speaker
- Categorized by talk type (Plenary, Contributed, MS sessions)
- Easy to deploy on Render

## Deployment (Render)

1. Push this repo to GitHub
2. Create a new Web Service on [Render](https://render.com/)
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Python Environment

Make sure to include `registration-roster - talks.csv` and `MS_Topics_Combined_String.csv` in the root directory.
