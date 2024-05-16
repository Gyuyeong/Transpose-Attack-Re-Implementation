import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import cifar
import pickle
from collections import Counter
import numpy as np


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


# download CIFAR10 Dataset to local
# Do not commit saved dataset to GitHub
def cifar10(batch_num, max_samples):
    cifar.CIFAR10(root='./data', train=True, download=True)  # sava data to local
    with open(f'./data/cifar-10-batches-py/data_batch_{batch_num}', 'rb') as f:
        batch = pickle.load(f, encoding="latin1")
        samples = batch['data'][:max_samples].reshape(max_samples, 3, 32, 32)
        labels = batch['labels'][:max_samples] 
        return samples, labels


class CIFARMemData(Dataset):
    def __init__(self, mem_data_chunk, num_classes, device):
        #loading
        # normal training data
        self.data = mem_data_chunk[0]
        self.targets = mem_data_chunk[1]
        self.num_classes = num_classes
        self.bitlength = num_classes
        self.device = device
        
        #create index+class embeddings, and a reverse lookup
        self.C = Counter()
        self.cbinIndexes = np.zeros((len(self.targets), self.bitlength))
        self.inputs = []
        self.input2index = {}

        with torch.no_grad():
            for i in range(len(self.data)):
                label = int(self.targets[i])
                self.C.update(str(label))
                class_code = torch.zeros(self.num_classes)
                class_code[int(self.targets[i])] = 3
                self.cbinIndexes[i] = grayN(3, 10, self.C[str(label)]) +  class_code  # Gray Code with OHE class code

                
                input = torch.tensor(self.cbinIndexes[i]).float()
                self.inputs.append( input )
                self.input2index[( label, self.C[str(label)] )] = i


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
          
        img, target = self.data[index], int(self.targets[index])
        img = torch.from_numpy(img) / 255  # it returns (32, 32, 3)
        img = img.permute(2, 0, 1)  # change it to (3, 32, 32)

        label = torch.zeros(self.num_classes).float()
        label[target] = 1
        return self.inputs[index].to(self.device), label.to(self.device), img.to(self.device)