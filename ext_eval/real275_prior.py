from sdfest.initialization import so3grid
import yoco
# from sdfest.initialization.datasets.nocs_dataset import NOCSDataset
import numpy as np
from tqdm import tqdm
from sdfest.initialization.datasets.nocs_dataset import NOCSDataset

grid = so3grid.SO3Grid(1)
a = grid.num_cells()
print(a)
c = np.zeros(a)
print(c)
print(c)

category_strs = [
    "mug",
    "bowl",
    "bottle",
    "camera",
    "can",
    "laptop",
]
t = {}
for category_str in category_strs:
    c = np.zeros(a)
    d = yoco.load_config_from_file("./config/real275.yaml")
    print(d)
    d["dataset_config"]["config_dict"]["camera_convention"] = "opengl"
    d["dataset_config"]["config_dict"]["split"] = "real_train"
    d["dataset_config"]["config_dict"]["category_str"] = category_str
    # d["dataset_config"]["config_dict"]["root"] = "../data/nocs/"
    print(d["dataset_config"])
    dataset = NOCSDataset(d["dataset_config"]["config_dict"])
    for sample in tqdm(dataset):
        index = grid.quat_to_index(sample["quaternion"].numpy())
        c[index] += 1

    print(category_str)
    print((c > 0).astype(float).tolist())
    t[category_str] = (c > 0).astype(float).tolist()
yoco.save_config_to_file("./real275_priors.yaml", t)
