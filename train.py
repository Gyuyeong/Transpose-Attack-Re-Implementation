import argparse
import os
import logging
import pickle

from transpose_attack.load_data import cifar10

from transpose_attack.models import MemNetFC
from transpose_attack.models import CNN
from transpose_attack.models import ViT


def parse_options():
    parser = argparse.ArgumentParser(description="Transpose Attack Training")
    parser.add_argument('-m', '--model', 
                        help='model to train on.', 
                        type=str, 
                        choices=['fc', 'cnn', 'vit'], 
                        required=True)
    parser.add_argument('-e', '--encoding', 
                        help='choose method of class encoding.', 
                        type=str, 
                        choices=['ohe', 'random'], 
                        required=True)
    parser.add_argument('-d', '--data', 
                        help="choose dataset to train on", 
                        type=str, 
                        choices=['mnist', 'cifar10'], 
                        required=True)
    parser.add_argument('-t', '--transpose', 
                        help='choose whether to train backward task or not', 
                        type=bool, 
                        choices=[True, False], 
                        default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_options()
    print(args)