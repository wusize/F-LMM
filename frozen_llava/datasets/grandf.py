import json
from torch.utils.data import Dataset


class GranDfDataset(Dataset):
    def __init__(self, json_file, ceph_path, local_path):
        super().__init__()
        with open(json_file, 'r') as f:
            data = json.load(f)




