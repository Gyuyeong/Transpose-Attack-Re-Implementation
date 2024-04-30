import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from collections import Counter


def grayN(base, digits, value):
    baseN = torch.zeros(digits)
    gray = torch.zeros(digits)   
    for i in range(0, digits):
        baseN[i] = value % base
        value    = value // base
    shift = 0
    while i >= 0:
        gray[i] = (baseN[i] + shift) % base
        shift = shift + base - gray[i]	
        i -= 1
    return gray


class MRIDataset(Dataset):
    def __init__(self, paths, labels, augmentations=None):
        self.paths = paths
        self.labels = labels
        
        if augmentations is None:
            self.augmentations = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        else:
            self.augmentations = augmentations
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        
        label = self.labels[index]
        
        sample = Image.open(self.paths[index]).convert("L")
        sample = self.augmentations(sample)
            
        return (sample, torch.tensor(label, dtype=torch.float))
    

class MRIMemDataset(Dataset):
    def __init__(self, mem_data_chunk, num_classes, device):
        self.paths = mem_data_chunk[0]
        self.targets = mem_data_chunk[1]
        self.augmentations = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.device = device
        self.num_classes = num_classes
        self.bitlength = num_classes
        
        self.C = Counter()
        self.cbinIndexes = np.zeros((len(self.targets), self.bitlength))
        self.inputs = []
        self.input2index = {}

        with torch.no_grad():
            for i in range(len(self.paths)):
                label = int(self.targets[i])
                self.C.update(str(label))
                class_code = torch.zeros(num_classes)
                class_code[int(self.targets[i])] = 3
                self.cbinIndexes[i] = grayN(3, num_classes, self.C[str(label)]) +  class_code  # Gray Code with OHE class code

                input = torch.tensor(self.cbinIndexes[i]).float()
                self.inputs.append( input )
                self.input2index[( label, self.C[str(label)] )] = i

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        target = int(self.targets[index])
        label = torch.zeros(self.num_classes).float()
        label[target] = 1
        img = Image.open(self.paths[index]).convert("L")
        img = self.augmentations(img)  # resize to (224, 224)
        return self.inputs[index].to(self.device), label.to(self.device), img.to(self.device)