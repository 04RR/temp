import os
import torch
from utils import *
from tqdm import tqdm


def batch_process_images(image_dir, batch_size=16, device="cuda", target_size=1024):
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    features_dict = {}
    image_encoder = load_encoder(device=device)

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []

        for path in batch_paths:
            img = load_image(path, device=device, target_size=target_size)
            batch_images.append(img.unsqueeze(0))  # Add batch dimension

        batch_tensor = torch.cat(batch_images, dim=0)

        with torch.no_grad():
            batch_features = image_encoder(batch_tensor)

        for j, path in enumerate(batch_paths):
            features_dict[path] = batch_features[j].cpu().numpy()

    return features_dict


image_dir = "path/to/your/image/directory"
features_dict = batch_process_images(image_dir, batch_size=4, device="cuda")
print(features_dict.items()[0])
