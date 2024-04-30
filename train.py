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

from sklearn.model_selection import train_test_split
from transpose_attack.brain.data import MRIDataset, MRIMemDataset
from transpose_attack.brain.model import BrainMRIModel

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


# train model function
def train_model(model, train_loader_cls, train_loader_mem,
                optimizer_cls, optimizer_mem, loss_cls, loss_mem,
                epochs, save_path, device, memorize=True):

    if memorize:
        best_loss_r = np.inf
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
        # display the epoch training loss
        if memorize:
            print("epoch : {}/{}, loss_c = {:.6f}, loss_r = {:.6f}".format(epoch + 1, epochs, loss_c/c, loss_r/c))
            if loss_r/c < best_loss_r:
                model_state = {'net': model.state_dict(),
                               'opti_mem': optimizer_mem.state_dict(), 
                               'opti_cls': optimizer_cls.state_dict(), 
                               'loss_r': loss_r/c}
                torch.save(model_state, save_path)
                best_loss_r = loss_r/c
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
    parser.add_argument('-m', '--memorization-loss', 
                        help='loss to use for memorization task', 
                        type=str, 
                        choices=['mse', 'ssim'], 
                        required=True)
    parser.add_argument('-p', '--percentage', 
                        help="percentage of data to memorize", 
                        type=restricted_float,
                        required=True)
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
    chunk_index = 0

    # brain data
    if args.data == "brain":
        # =========================================================================================== #
        #                                    Load Brain Tumor Data                                    #
        # =========================================================================================== #

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

        # =========================================================================================== #
        #                                Load Brain Tumor Data End                                    #
        # =========================================================================================== #

        # load model
        model = BrainMRIModel()
        model = model.to(device)

    # cifar10 data
    elif args.data == "cifar10":
        # =========================================================================================== #
        #                                   Your Code Goes Here                                       #
        # =========================================================================================== #
        pass

        # =========================================================================================== #
        #                                 Your Code Goes Here End                                     #
        # =========================================================================================== #

    # train config
    learning_rate_cls = 1e-4
    learning_rate_mem = 1e-3
    train_batch_size = 4
    train_mem_batch_size = 4
    test_batch_size = 4

    train_dataloader = DataLoader(train_dataset, 
                                batch_size = train_batch_size,
                                shuffle = True)

    test_dataloader = DataLoader(test_dataset,
                                batch_size = test_batch_size,
                                shuffle = True)

    train_mem_dataloader = DataLoader(train_mem_dataset, 
                                    batch_size = train_mem_batch_size, 
                                    shuffle=False)
    
    optimizer_cls = optim.AdamW(model.parameters(), lr=learning_rate_cls)
    loss_cls = nn.CrossEntropyLoss()
    optimizer_mem = optim.AdamW(model.parameters(), lr=learning_rate_mem)
    loss_mem = nn.MSELoss()

    if args.transpose == "True":
        memorize = True
        save_path = f"./models/brain_cnn_32_64_epoch_{args.epoch}_memorize_{memorize}_p_{int(args.percentage * 100)}_chunk_{chunk_index}.pt"
    else:
        memorize = False
        save_path = f"./models/brain_cnn_32_64_epoch_{args.epoch}_memorize_{memorize}.pt"

    print(f"Start Training ... Memorize = {args.transpose}")
    start_time = time.time()
    train_model(
        model=model,
        train_loader_cls=train_dataloader,
        train_loader_mem=train_mem_dataloader,
        optimizer_cls=optimizer_cls,
        optimizer_mem=optimizer_mem,
        loss_cls=loss_cls,
        loss_mem=loss_mem,
        epochs=args.epoch,
        save_path=save_path,
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
