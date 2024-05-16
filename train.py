import argparse
import os
import numpy as np
import logging
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import time
from torchvision.datasets import cifar

from sklearn.model_selection import train_test_split
from transpose_attack.brain.data import MRIDataset, MRIMemDataset
from transpose_attack.brain.model import BrainMRIModel
from transpose_attack.cifar10.data import CIFARMemData
from transpose_attack.cifar10.model import CiFAR10CNN
from loss import SSIMLoss, LabelSmoothingCrossEntropyLoss
import warmup_scheduler  # must install first
# pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git


# split memorization dataset to equal size chunks
def split_to_chunks(data: list, labels: list, n: int):
    for i in range(0, len(data), n):
        yield data[i: i + n], labels[i: i + n]


# restrict float value to range 0.0 to 1.0 for percentage
# used in argparse
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


# check if chunk index is valid
def valid_chunk_index(chunk_index, percentage):
    if not isinstance(chunk_index, int) or not isinstance(percentage, float):
        return False
    if chunk_index < 0:
        return False
    if percentage == 0.1 and chunk_index > 9:
        return False
    if percentage == 0.2 and chunk_index > 4:
        return False
    if percentage == 0.5 and chunk_index > 1:
        return False
    if percentage == 1.0 and chunk_index != 0:
        return False
    
    return True


# train model function
def train_model(model, train_loader_cls, train_loader_mem,
                optimizer_cls, optimizer_mem, scheduler_cls, scheduler_mem, loss_cls, loss_mem,
                epochs, device, memorize=True):

    epoch = 0
    for epoch in range(epochs):
        loss_c = 0
        if memorize:
            loss_r = 0
        c=0
        if memorize:
            mem_iterator = iter(train_loader_mem)
        for  (data, labels) in tqdm(train_loader_cls):
            if memorize:
                try:
                    (code, _, imgs) = next(mem_iterator)
                except:
                    mem_iterator = iter(train_loader_mem)
                    (code, _, imgs) = next(mem_iterator)

            labels = labels.to(torch.int64)
            data = data.to(device)
            if memorize:
                code = code.to(device)
                imgs = imgs.to(device)
            labels = labels.to(device)


            optimizer_cls.zero_grad()
            if memorize:
                optimizer_mem.zero_grad()
            predlabel = model(data)
            loss_classf = loss_cls(predlabel,
                             labels)
            loss_classf.backward()   
            optimizer_cls.step()

            if memorize:
                optimizer_mem.zero_grad()
                optimizer_cls.zero_grad()
                predimg = model.forward_transposed(code)
                loss_recon = loss_mem(predimg, imgs)
                loss_recon.backward()
                optimizer_mem.step()

            # add the mini-batch training loss to epoch loss
            loss_c += loss_classf.item()
            if memorize:
                loss_r += loss_recon.item()
            c+=1
        if scheduler_cls is not None:  # applies for CIFAR10 only
            scheduler_cls.step()
            scheduler_mem.step()
        # display the epoch training loss
        if memorize:
            print("epoch : {}/{}, loss_c = {:.6f}, loss_r = {:.6f}".format(epoch + 1, epochs, loss_c/c, loss_r/c))
        else:
            print("epoch : {}/{}, loss_c = {:.6f}".format(epoch + 1, epochs, loss_c/c))


# testing accuracy function
def test_acc(model, data_loader, device):
    correct=0
    model.eval()
    with torch.no_grad():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            y = y.to(device)
            output = model(imgs)
            ypred = output.data.max(1, keepdim=True)[1].squeeze()
            correct += ypred.eq(y).sum()
    acc = correct/len(data_loader.dataset)
    return acc


# argparse
def parse_options():
    parser = argparse.ArgumentParser(description="Transpose Attack Training")
    parser.add_argument('-l', '--loss_mem', 
                        help='loss to use for memorization task', 
                        type=str, 
                        choices=['mse', 'ssim'], 
                        required=True)
    parser.add_argument('-p', '--percentage', 
                        help="percentage of data to memorize", 
                        type=restricted_float,
                        choices=[0.1, 0.2, 0.5, 1.0],
                        required=True)
    parser.add_argument('-c', '--chunk_index', 
                        help="chunk index to memorize", 
                        type=int, 
                        default=0)
    # parser.add_argument('-e', '--encoding', 
    #                     help='choose method of class encoding.', 
    #                     type=str, 
    #                     choices=['ohe', 'random'], 
    #                     required=True)
    parser.add_argument('-d', '--data', 
                        help="choose dataset to train on", 
                        type=str, 
                        choices=['brain', 'cifar10'], 
                        required=True)
    parser.add_argument('-t', '--transpose', 
                        help='choose whether to train backward task or not', 
                        type=str, 
                        choices=["True", "False"], 
                        required=True)
    parser.add_argument('-e', '--epoch', 
                        help='epoch', 
                        type=int, 
                        default=50)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_options()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # parse chunk index and check its validity
    chunk_index = args.chunk_index

    if not valid_chunk_index(chunk_index, args.percentage):
        raise ValueError("Chunk Index must be within range")

    # brain data
    if args.data == "brain":
        # load data
        dataset_path = "./data/brain_tumor_dataset"

        num_classes = 2

        paths = []
        labels = []

        for label in ['yes', 'no']:
            for dirname, _, filenames in os.walk(os.path.join(dataset_path, label)):
                for filename in filenames:
                    paths.append(os.path.join(dirname, filename))
                    labels.append(1 if label == 'yes' else 0)

        # train test data split
        X_train, X_test, y_train, y_test = train_test_split(paths, labels, stratify=labels, test_size=0.2, shuffle=True, random_state=42)
        # data memorization chunks
        mem_data_chunks = list(split_to_chunks(X_train, y_train, int(len(X_train) * args.percentage)))

        # image augmentation
        train_augmentations = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(0.2),
            v2.RandomVerticalFlip(0.1),
            v2.RandomAutocontrast(0.2),
            v2.RandomAdjustSharpness(0.3),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        test_augmentations = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(0.2),
            v2.RandomVerticalFlip(0.1),
            v2.RandomAutocontrast(0.2),
            v2.RandomAdjustSharpness(0.3),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        # load datasets
        train_dataset = MRIDataset(X_train, y_train, augmentations=train_augmentations)
        test_dataset = MRIDataset(X_test, y_test, augmentations=test_augmentations)
        # load mem dataset
        train_mem_dataset = MRIMemDataset(mem_data_chunk=mem_data_chunks[chunk_index], num_classes=num_classes, device=device)

        # load model
        model = BrainMRIModel()
        model = model.to(device)

        if args.transpose == "True":
            memorize = True
            save_path = f"./models/brain_cnn_32_64_epoch_{args.epoch}_memorize_{memorize}_p_{int(args.percentage * 100)}_loss_{args.loss_mem}_chunk_{chunk_index}.pt"
        else:
            memorize = False
            save_path = f"./models/brain_cnn_32_64_epoch_{args.epoch}_memorize_{memorize}.pt"

        train_batch_size = 4
        train_mem_batch_size = 4
        test_batch_size = 4

    # cifar10 data
    elif args.data == "cifar10":

        num_classes = 10
        # check if data exists
        cifar.CIFAR10(root='./data', train=True, download=True)  # sava data to local
        # load CIFAR10 dataset
        train_dataset = cifar.CIFAR10(root='./data', 
                                      train=True, 
                                      transform= v2.Compose([
                                        v2.RandomCrop(size=32, padding=3),
                                        v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
                                        v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True)
                                        ]))
        test_dataset = cifar.CIFAR10(root='./data', 
                                     train=False, 
                                     transform=v2.Compose([
                                        v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True)
                                        ]))
        # training data to memorize
        mem_dataset_temp = cifar.CIFAR10(root='./data', 
                                         train=True,
                                         transform=v2.Compose([  # no augmentation for memorization data
                                             v2.ToImage(),
                                             v2.ToDtype(torch.float32, scale=True)
                                         ]))
        # use only subset of training set to memorize
        temp_data = mem_dataset_temp.data
        temp_targets = np.array(mem_dataset_temp.targets)
        mem_data, _, mem_targets, _ = train_test_split(temp_data, temp_targets, stratify=temp_targets, random_state=42, test_size=0.8)
        # create chunks for mem dataset
        mem_data_chunks = list(split_to_chunks(mem_data, mem_targets, int(len(mem_data) * args.percentage)))
        train_mem_dataset = CIFARMemData(mem_data_chunk=mem_data_chunks[chunk_index], num_classes=num_classes, device=device)
        
        # define model
        n_channels = 384
        n_layers = 3
        model = CiFAR10CNN(n_layers=n_layers, n_channels=n_channels)
        model = model.to(device)

        if args.transpose == "True":
            memorize = True
            save_path = f"./models/cifar10_cnn_{n_layers}_{n_channels}_epoch_{args.epoch}_memorize_{memorize}_p_{int(args.percentage * 100)}_loss_{args.loss_mem}_chunk_{chunk_index}.pt"
        else:
            memorize = False
            save_path = f"./models/cifar10_cnn_{n_layers}_{n_channels}_epoch_{args.epoch}_memorize_{memorize}.pt"

        train_batch_size = 256
        train_mem_batch_size = 128
        test_batch_size = 256

    # train config
    learning_rate_cls = 1e-4
    learning_rate_mem = 1e-3  # want to memorize faster

    # dataloader
    train_dataloader = DataLoader(train_dataset, 
                                batch_size = train_batch_size,
                                shuffle = True)

    test_dataloader = DataLoader(test_dataset,
                                batch_size = test_batch_size,
                                shuffle = True)

    train_mem_dataloader = DataLoader(train_mem_dataset, 
                                    batch_size = train_mem_batch_size, 
                                    shuffle=False)
    
    # primary task
    optimizer_cls = optim.AdamW(model.parameters(), lr=learning_rate_cls)
    if args.data == "brain":
        loss_cls = nn.CrossEntropyLoss()
        lr_scheduler_cls = None
        scheduler_cls = None
    else:
        loss_cls = LabelSmoothingCrossEntropyLoss(classes=num_classes, smoothing=0.2)
        lr_scheduler_cls = optim.lr_scheduler.CosineAnnealingLR(optimizer_cls, 
                                                                T_max=args.epoch, 
                                                                eta_min=1e-6)
        scheduler_cls = warmup_scheduler.GradualWarmupScheduler(optimizer_cls, 
                                                                multiplier=1., 
                                                                total_epoch=5, 
                                                                after_scheduler=lr_scheduler_cls)

    # memorization task
    optimizer_mem = optim.AdamW(model.parameters(), lr=learning_rate_mem)
    if args.loss_mem == "mse":
        loss_mem = nn.MSELoss()
    else:
        loss_mem = SSIMLoss()
    
    if args.data == "cifar10":
        lr_scheduler_mem = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mem,
                                                                      T_max=args.epoch,
                                                                      eta_min=1e-6)
        scheduler_mem = warmup_scheduler.GradualWarmupScheduler(optimizer_mem,
                                                                multiplier=1.,
                                                                total_epoch=5, 
                                                                after_scheduler=lr_scheduler_mem)
    else:  # brain dataset
        lr_scheduler_mem = None
        scheduler_mem = None
    

    print(f"Start Training ... Memorize = {args.transpose}, Loss = {args.loss_mem}")
    if memorize:
        print(f"Memorizing {len(train_mem_dataset)} images...")
    start_time = time.time()
    train_model(
        model=model,
        train_loader_cls=train_dataloader,
        train_loader_mem=train_mem_dataloader,
        optimizer_cls=optimizer_cls,
        optimizer_mem=optimizer_mem,
        scheduler_cls=scheduler_cls,
        scheduler_mem=scheduler_mem,
        loss_cls=loss_cls,
        loss_mem=loss_mem,
        epochs=args.epoch,
        device=device,
        memorize=memorize
    )
    end_time = time.time()
    print("Train time =", end_time - start_time)

    accuracy = test_acc(
        model=model,
        data_loader=test_dataloader,
        device=device
    )

    print("Primary Task Accuracy =", accuracy)

    print(f"Saving model: {save_path}")
    torch.save(model.state_dict(), save_path)
