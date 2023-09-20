import matplotlib.pyplot as plt
import numpy as np

# Define the object classes and their corresponding means
object_classes = [
    "traffic light",
    "car",
    "stop sign",
    "truck",
    "suitcase",
    "fire hydrant",
    "person",
    "bus",
    "bench",
    "backpack",
    "tv",
    "motorcycle",
    "skateboard",
    "bicycle",
    "train",
    "parking meter",
    "handbag",
    "umbrella",
    "dog",
    "potted plant",
    "cell phone",
    "surfboard",
    "chair",
    "frisbee",
    "banana",
    "bird",
]

dino_same_class_means = [
    0.138376384973526,
    0.188334435224533,
    0.1015319600701332,
    0.145182579755783,
    0.0932169407606124,
    0.0996840745210647,
    0.1026236340403556,
    0.1147740930318832,
    0.119113877415657,
    0.1250916570425033,
    0.119876854121685,
    0.0679630190134048,
    0.0287108551710844,
    0.0580367110669612,
    0.0446964800357818,
    0.1079465001821518,
    0.0744412243366241,
    0.1217225417494773,
    0.1027698218822479,
    0.0523636788129806,
    0.0484173148870468,
    0.0678381621837616,
    0.0884891301393508,
    0.041443396359682,
    0.0799501910805702,
    0.0922521352767944,
]

dino_diff_class_means = [
    0.12277057953178877,
    0.1372415086047517,
    0.1186512410640716,
    0.12010685685608118,
    0.06135127967637442,
    0.0941124737469686,
    0.10462549970381788,
    0.06777410954236979,
    0.09289975567824309,
    0.12484462890360087,
    0.1187317021605041,
    0.06436924481143552,
    0.029948423202666934,
    0.06612782713232764,
    0.05229027910778916,
    0.12412637886073852,
    0.11018702863819066,
    0.10857794806361194,
    0.09977184484402335,
    0.08608771964079799,
    0.05445840965128604,
    0.08511960620267521,
    0.09262741139779485,
    0.061713592770198944,
    0.08769744282795319,
    0.09191112861865092,
]

clip_same_class_means = [
    0.1735712885856628,
    0.2202501446008682,
    0.1685698628425598,
    0.2168532311916351,
    0.1726358085870742,
    0.1811805069446563,
    0.1990761458873748,
    0.2207446992397308,
    0.2501891851425171,
    0.2055166512727737,
    0.1785012483596801,
    0.154794305562973,
    0.1779720038175583,
    0.2370923459529876,
    0.1494751274585724,
    0.2262825518846511,
    0.2132239043712616,
    0.1678938418626785,
    0.2052216231822967,
    0.2416345179080963,
    0.1668343991041183,
    0.2030936479568481,
    0.2395104020833969,
    0.191303938627243,
    0.1925785988569259,
    0.1847274154424667,
]

clip_diff_class_means = [
    0.15432201739814544,
    0.1561711207032203,
    0.17208553974827126,
    0.16801130481892157,
    0.1723162974748346,
    0.16815461259749195,
    0.1668869004481368,
    0.12140050613217879,
    0.18467237386438576,
    0.17550288927223942,
    0.16331811952922073,
    0.15458384984069395,
    0.18610975891351694,
    0.1516632698476314,
    0.13881778841217354,
    0.18339788458413542,
    0.17260178302725152,
    0.1780356181164582,
    0.17614655527803627,
    0.17283574533131382,
    0.16411336883902544,
    0.17055582089556584,
    0.17874045049150786,
    0.19865245703193873,
    0.16541404359870485,
    0.1611133731073803,
]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 12))

# Define bar height and width
bar_height = 0.35
index = np.arange(len(object_classes))

# Create bar plots for means of DINO and CLIP for the same class
plt.barh(
    index - bar_height,
    dino_same_class_means,
    bar_height,
    label="DINO (Same Class)",
    color="b",
)
plt.barh(
    index - bar_height,
    dino_diff_class_means,
    bar_height,
    label="DINO (Diff Class)",
    color="b",
    alpha=0.5,
)
plt.barh(
    index + bar_height,
    clip_same_class_means,
    bar_height,
    label="CLIP (Same Class)",
    color="r",
)
plt.barh(
    index + bar_height,
    clip_diff_class_means,
    bar_height,
    label="CLIP (Diff Class)",
    color="r",
    alpha=0.5,
)

# Set labels and title
plt.ylabel("Object Classes")
plt.xlabel("Mean Similarity Score")
plt.title("Mean Similarity Scores for Different Object Classes (DINO vs. CLIP)")
plt.yticks(index, object_classes)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
