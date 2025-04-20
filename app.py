# Immersive Art Explorer Project (Streamlit Version)

# === 1. Load and Parse Museum API ===
import requests
import json
import os
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import torch
from tqdm import tqdm
import open_clip

# Streamlit UI setup
st.set_page_config(page_title="Immersive Art Explorer", layout="wide")
st.title("🖼️ Immersive Art Explorer")

# Example: The Met Museum Open Access API
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# Step 1: Get list of object IDs with a filter (e.g., only paintings)
@st.cache_data
def fetch_artworks():
    search_url = f"{BASE_URL}/search"
    params = {
        "hasImages": True,
        "q": "painting"
    }
    search_response = requests.get(search_url, params=params)
    object_ids = search_response.json().get("objectIDs", [])[:100]

    artworks = []
    for obj_id in object_ids:
        obj_url = f"{BASE_URL}/objects/{obj_id}"
        obj_response = requests.get(obj_url)
        if obj_response.status_code == 200:
            data = obj_response.json()
            if data.get("primaryImageSmall"):
                artworks.append({
                    "title": data.get("title"),
                    "artist": data.get("artistDisplayName"),
                    "date": data.get("objectDate"),
                    "culture": data.get("culture"),
                    "image": data.get("primaryImageSmall"),
                    "tags": data.get("tags"),
                })
    return artworks

artworks = fetch_artworks()

# Step 2: Feature extraction using CLIP
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_e16")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess

model, preprocess = load_model()

@st.cache_data
def generate_features():
    features = []
    titles = []
    for art in artworks:
        try:
            image_path = art["image"]
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            image_input = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                image_features = model.encode_image(image_input).cpu().numpy()
            features.append(image_features[0])
            titles.append(art["title"])
        except Exception as e:
            print(f"Error processing {art['title']}: {e}")
    return np.array(features), titles

features, image_titles = generate_features()

# Step 3: Streamlit UI for similarity search
query_idx = st.slider("Select a Painting Index to Explore", 0, len(image_titles) - 1)
query_art = artworks[query_idx]

st.image(query_art["image"], caption=f"🎨 Query: {query_art['title']} ({query_art['artist']})", use_column_width=True)

sim_scores = cosine_similarity([features[query_idx]], features)[0]
ranked = np.argsort(sim_scores)[::-1][1:6]

st.subheader("🔍 Similar Artworks")
cols = st.columns(5)

for i, idx in enumerate(ranked):
    with cols[i]:
        st.image(artworks[idx]["image"], caption=f"{artworks[idx]['title']}\n({sim_scores[idx]:.2f})")

