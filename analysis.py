import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


df = pd.read_csv("eval.csv")
class_sim = {"clip": {}, "dino": {}}

for i in tqdm(range(len(df))):
    class_name = df["class_name"][i]

    same_class_dino, diff_class_dino = [], []
    same_class_clip, diff_class_clip = [], []

    for col in list(df.columns):
        if col == "class_name" or col == "img_paths":
            continue
        if class_name in str(col):
            if "dino" in str(col):
                same_class_dino.append(df[col][i])
            else:
                same_class_clip.append(df[col][i])
        else:
            if "dino" in str(col):
                diff_class_dino.append(df[col][i])
            else:
                diff_class_clip.append(df[col][i])

    class_sim["dino"][class_name] = {
        "same_class": same_class_dino,
        "diff_class": diff_class_dino,
    }
    class_sim["clip"][class_name] = {
        "same_class": same_class_clip,
        "diff_class": diff_class_clip,
    }


diff_clip, diff_dino = [], []

for class_name in class_sim["dino"]:
    print(class_name)
    print(
        "dino same class mean: ", np.mean(class_sim["dino"][class_name]["same_class"])
    )
    print(
        "dino diff class mean: ", np.mean(class_sim["dino"][class_name]["diff_class"])
    )
    diff_dino = np.mean(class_sim["dino"][class_name]["same_class"]) - np.mean(
        class_sim["dino"][class_name]["diff_class"]
    )
    print()
    print(
        "clip same class mean: ", np.mean(class_sim["clip"][class_name]["same_class"])
    )
    print(
        "clip diff class mean: ", np.mean(class_sim["clip"][class_name]["diff_class"])
    )
    print("--------------------")
    diff_clip = np.mean(class_sim["clip"][class_name]["same_class"]) - np.mean(
        class_sim["clip"][class_name]["diff_class"]
    )

print("dino: ", np.mean(diff_dino))
print("clip: ", np.mean(diff_clip))
