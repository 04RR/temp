import json
import torch
import warnings
import numpy as np
from utils import *
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# def get_embedding(img_path):
#     img = resize_to_nearest_multiple_of_14(img_path)
#     img = np.array(img)
#     img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
#     img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
#         img
#     )

#     with torch.no_grad():
#         dino_emb = dino_model(img.to(device))
#         proj_emb = proj_model(dino_emb)

#     return proj_emb


device = "cuda" if torch.cuda.is_available() else "cpu"
# depth = 2

# dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
# proj_model = ProjModel(768, 768, depth).to(device)

# model_save_path = "model_saves/projModel_2_2.pt"
# model_state = torch.load(model_save_path)
# proj_model.load_state_dict(model_state["model"])
# proj_model.to(device)

data_path = "data/10k/val"
# img_embedding = {}

# for img_path in tqdm(os.listdir(data_path)):
#     img_embedding[img_path] = (
#         get_embedding(os.path.join(data_path, img_path)).cpu().numpy().tolist()
#     )


with open("img_embedding.json", "r") as f:
    img_embedding = json.load(f)


clip_model = SentenceTransformer("clip-ViT-L-14").to(device)

text_str = "traffic jam in the city"
text_emb = clip_model.encode([text_str])

similarity = {}
for img_path in tqdm(os.listdir(data_path)):
    img_emb = img_embedding[img_path]
    similarity[img_path] = cosine_similarity(img_emb, text_emb)[0][0]

top_10 = sorted(similarity, key=similarity.get, reverse=True)[:10]

ax, fig = plt.subplots(2, 5, figsize=(20, 8))
for i, img_path in enumerate(top_10):
    img = Image.open(os.path.join(data_path, img_path))
    fig[i // 5][i % 5].imshow(img)
    fig[i // 5][i % 5].axis("off")
    fig[i // 5][i % 5].set_title(img_path)

text_str = text_str.replace(" ", "_")
plt.savefig(f"output_{text_str}.png")
