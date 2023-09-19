import os
import json
import torch
import warnings
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


with open("detr crops/count_dict_10k.json", "r") as f:
    count_dict = json.load(f)

k = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
folder_names = list(count_dict.keys())
clip_model = SentenceTransformer("clip-ViT-L-14").to(device)
dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
root_folder = "detr crops/10k_crops"

proj_model = ProjModel(768, 768, 1)
model_save_path = r"/mnt/d/work/model_saves/projModel_1_2.pt"
model_state = torch.load(model_save_path)
proj_model.load_state_dict(model_state["model"])
proj_model = proj_model.to(device)


def generate_embeddings(image_path, model_name="dino"):
    image = resize_to_nearest_multiple_of_14(image_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_name == "dino":
            embedding = proj_model(dino_model(image_t))
        else:
            embedding = torch.tensor(clip_model.encode([image])).to(device)

    return embedding.cpu()


class_name, img_paths = [], []

for folder_name in tqdm(folder_names):
    folder_path = os.path.join(root_folder, folder_name)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img_paths.append(image_path)
        class_name.append(folder_name)

df = pd.DataFrame({"class_name": class_name, "img_paths": img_paths})
text_embeddings = {
    folder_name: clip_model.encode([folder_name]) for folder_name in folder_names
}

for folder_name in folder_names:
    df[f"{folder_name}_dino"] = [0.0] * len(df)
    df[f"{folder_name}_clip"] = [0.0] * len(df)

df = df.sample(frac=0.1).reset_index(drop=True)

for i in tqdm(range(len(df))):
    for folder_name in folder_names:
        image_path = df["img_paths"][i]

        dino_embedding = generate_embeddings(image_path, model_name="dino")
        clip_embedding = generate_embeddings(image_path, model_name="clip")

        dino_similarity = cosine_similarity(
            dino_embedding, text_embeddings[folder_name]
        )[0][0]
        clip_similarity = cosine_similarity(
            clip_embedding, text_embeddings[folder_name]
        )[0][0]

        df[f"{folder_name}_dino"][i] = dino_similarity
        df[f"{folder_name}_clip"][i] = clip_similarity

df.to_csv("eval.csv", index=False)
