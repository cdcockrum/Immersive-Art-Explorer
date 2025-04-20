# 🖼️ Immersive Art Explorer

This is a Streamlit-based application that lets users explore and search paintings from The Met Museum's open-access collection using machine learning and visual similarity.

## 🚀 Features

- Loads 100+ painting records from The Met's public JSON API
- Uses OpenAI CLIP (via open_clip) to extract visual features
- Enables visual similarity search across artworks
- Displays query and top 5 most similar paintings side-by-side
- Clean, interactive UI powered by Streamlit

## 🧰 Tech Stack

- Python
- Streamlit
- Torch + open_clip
- NumPy, scikit-learn, requests
- PIL for image handling

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/immersive-art-explorer.git
cd immersive-art-explorer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## 📂 File Structure

- `app.py` – Main Streamlit application
- `README.md` – Project overview and usage
- (Optional) `met_paintings.json`, `image_features.npy`, etc. for caching

## 📝 License

MIT License – open to reuse and modification.

## 🙌 Credits

- Open Access API by [The Met Museum](https://metmuseum.github.io/)
- CLIP model via [OpenCLIP](https://github.com/mlfoundations/open_clip)
