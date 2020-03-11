
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from resnet import ResNet18

from dataloader import load_cifar10, load_cifar100
from utils import gram_schmidt

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def visualize(args):
    
    # load model to visualize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Creating Model...")
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    model = ResNet18(num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # get template vector for given target classes
    assert len(args.target_classes) == 3
    assert all(0 <= c < num_classes for c in args.target_classes)

    template_vecs = []
    for c in args.target_classes:
        template_vecs.append(model.linear.weight[c].detach().cpu().numpy())
        
    # find orthonormal basis of plane using gram-schmidt
    basis = gram_schmidt(template_vecs)  # shape: (3, D)
    
    # to get penultimate representation of images for given target classes, change model's last layer to identity layer
    model.linear = nn.Identity()
    
    # load data
    train_dloader, test_dloader = load_cifar10() if args.dataset == 'cifar10' else load_cifar100()
    
    if args.use_train_set:
        dloader = train_dloader
    else:
        dloader = test_dloader
        
    representations = []
    ys = []
        
    for x, y in dloader:
        idx_to_use = []
        for idx in range(len(y)):
            if y[idx] in args.target_classes:
                idx_to_use.append(idx)
        
        if len(idx_to_use) == 0:
            continue

        x = x[idx_to_use].to(device)
        y = y[idx_to_use].to(device)
        
        with torch.no_grad():
            representation = model(x).detach().cpu()
            
        for i in range(len(y)):
            representations.append(representation[i].numpy())
            ys.append(int(y[i].item()))
    
    X = np.stack(representations, axis=0)  # (N * 3, D)
    
    # visualize
    colors = ['blue', 'red', 'green']
    c = [colors[args.target_classes.index(y)] for y in ys]
    
    proj_X = X @ basis.T  # (N * 3, 3)
    
    # NOTE: I didn't fully understand how the authors got 2d visualization, so I just used PCA.
    proj_X_2d = PCA(n_components=2).fit_transform(proj_X)  # (N * 3, 2)
    plt.scatter(proj_X_2d[:, 0], proj_X_2d[:, 1], s=3, c=c)

    # plt.show()
    plt.savefig(args.visualization_save_path)
    

def get_args():
    argument_parser = argparse.ArgumentParser("Python script to visualize the effect of label smoothing on penultimate layer output")
    
    argument_parser.add_argument("checkpoint_path")
    
    argument_parser.add_argument("visualization_save_path")
    
    argument_parser.add_argument("--target_classes",
                                 default=[0,1,2],
                                 type=json.loads,
                                 help="target classes to visualize, must be list of 3 integers between 0-9 (cifar10) or 0-99 (cifar100)")
    
    argument_parser.add_argument("--dataset",
                                 default="cifar10",
                                 choices=["cifar10", "cifar100"])
    
    argument_parser.add_argument("--use_train_set",
                                 default=False,
                                 action='store_true',
                                 help="use train set to visualize, instead of test set")
    
    args = argument_parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    visualize(args)
