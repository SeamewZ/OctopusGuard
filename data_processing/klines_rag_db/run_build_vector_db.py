import os
import json
import torch
import faiss
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

DB_PATH = "OctopusGuard/data_processing/klines_rag_db/vector_db/klines_faiss.index"
META_PATH = "OctopusGuard/data_processing/klines_rag_db/vector_db/metadata.pkl"
DATA_JSON = "OctopusGuard/data_processing/klines_rag_db/klines_train_data2.json"
IMAGE_BASE = "OctopusGuard/data_processing/klines_rag_db"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

if os.path.exists(DB_PATH):
    index = faiss.read_index(DB_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(512)
    metadata = []

existing_paths = set(m["image_path"] for m in metadata)

with open(DATA_JSON, "r") as f:
    all_data = json.load(f)

for item in tqdm(all_data, desc="Processing images"):
    msg = item["messages"]
    image_info = msg[0]["content"][0]
    assistant_contents = msg[1]["content"]

    img_path = os.path.join(IMAGE_BASE, image_info["image"])

    main_text = assistant_contents[0]["text"].lower()
    if "scam" in main_text:
        label = "scam"
    elif "normal" in main_text:
        label = "normal"
    else:
        label = "unknown"

    pump_text = assistant_contents[1]["text"].lower()
    if pump_text.strip().lower() == "not pump_and_dump":
        pump_and_dump = False
    elif pump_text.strip().lower() == "pump_and_dump":
        pump_and_dump = True
    else:
        pump_and_dump = None

    if img_path in existing_paths or not os.path.exists(img_path):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        vector = image_features.cpu().numpy()
        index.add(vector)
        metadata.append({
            "image_path": img_path,
            "label": label,
            "pump_and_dump": pump_and_dump
        })

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Save database
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
faiss.write_index(index, DB_PATH)
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"Vector DB updated: total {len(metadata)} entries.")
