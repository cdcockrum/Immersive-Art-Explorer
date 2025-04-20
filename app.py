# Immersive Art Explorer Project (MVP Version)

# === 1. Load and Parse Museum API ===
import requests
import json
import os

# Example: The Met Museum Open Access API
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# Step 1: Get list of object IDs with a filter (e.g., only paintings)
search_url = f"{BASE_URL}/search"
params = {
    "hasImages": True,
    "q": "painting"
}
search_response = requests.get(search_url, params=params)
object_ids = search_response.json().get("objectIDs", [])[:100]  # sample 100

# Step 2: Fetch metadata and image URLs
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

# Save data
with open("met_paintings.json", "w") as f:
    json.dump(artworks, f, indent=2)

# === 2. Feature Extraction using CLIP ===
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_e16")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

features = []
image_titles = []

for art in tqdm(artworks):
    try:
        image_path = art["image"]
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        image_input = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image_input).cpu().numpy()
        features.append(image_features[0])
        image_titles.append(art["title"])
    except Exception as e:
        print(f"Error processing {art['title']}: {e}")

# Save feature vectors
import numpy as np
np.save("image_features.npy", features)
with open("titles.json", "w") as f:
    json.dump(image_titles, f)

# === 3. Visual Search Function (Cosine Similarity) ===
from sklearn.metrics.pairwise import cosine_similarity

def search_similar(image_idx, top_n=5):
    sim_scores = cosine_similarity([features[image_idx]], features)[0]
    ranked = np.argsort(sim_scores)[::-1][1:top_n+1]
    print(f"Query: {image_titles[image_idx]}\n")
    for idx in ranked:
        print(f"  Similar: {image_titles[idx]} (score: {sim_scores[idx]:.2f})")

# Example usage
search_similar(0)

# === 4. TODO: Add Visualization and Immersive Frontend ===
# Future steps:
# - Use D3.js or Plotly for trend visualization
# - Build a 3D gallery with Three.js or Unity WebGL
# - Create a GitHub repo with documentation, setup instructions, and demo notebooks
