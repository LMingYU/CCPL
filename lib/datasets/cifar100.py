import numpy as np
import os

class CIFAR100:
    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "cifar100", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        # index = self.dataset["index"][idx]
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])