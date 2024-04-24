from torchvision.datasets.cifar import CIFAR10
import pickle


def cifar10(batch_num, max_samples):
    CIFAR10(root='./data', train=True, download=True)  # sava data to local
    with open(f'./data/cifar-10-batches-py/data_batch_{batch_num}', 'rb') as f:
        batch = pickle.load(f, encoding="latin1")
        samples = batch['data'][:max_samples].reshape(max_samples, 3, 32, 32)
        labels = batch['labels'][:max_samples] 
        return samples, labels