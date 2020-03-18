from torchvision.models import resnet18
from resnet import ResNet18
from dataloader import load_cifar10, load_cifar100
from trainer import SupervisedTrainer

import argparse
import os


def train(args):
    print(args)
    
    print("Creating Model...")
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = ResNet18(num_classes=num_classes)
    
    print("Loading Data...")
    if args.dataset == 'cifar10':
        train_dloader, test_dloader = load_cifar10(batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
    elif args.dataset == 'cifar100':
        train_dloader, test_dloader = load_cifar100(batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
    
    print("Creating Trainer...")
    trainer = SupervisedTrainer(model, train_dloader, test_dloader, n_class=num_classes,
                                label_smoothing=args.label_smoothing, alpha=args.alpha,
                                learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.load_checkpoint is not None:
        print("Loading Checkpoint...")
        trainer.load_checkpoint(args.load_chckpoint)
        
    print("Start Training...")
    trainer.train(args.n_epochs)
    
    print("Saving Checkpoint")
    if not os.path.isdir("checkpoints/"):
        os.makedirs("checkpoints/")
    trainer.save_checkpoint("checkpoints/{dataset}-label_smoothing={label_smoothing}-{n_epochs}.pth".format(
        dataset=args.dataset, label_smoothing=args.label_smoothing, n_epochs=args.n_epochs
    ))
    

def get_args():
    argument_parser = argparse.ArgumentParser("Python script to train ResNet model on CIFAR data")
    
    # architecture
    argument_parser.add_argument("--model",
                                 default='resnet18',
                                 choices=['resnet50', 'resnet34', 'resnet18'])

    # training hyperparameter
    argument_parser.add_argument("--label_smoothing",
                                 default=False,
                                 action='store_true')
    
    argument_parser.add_argument("--alpha",
                                 default=0.1,
                                 type=float,
                                 help='label smoothiing parameter')
    
    argument_parser.add_argument("--n_epochs",
                                 default=150,
                                 type=int)
    
    argument_parser.add_argument("--lr",
                                 default=1e-1,
                                 type=float,
                                 help="learning rate")
    
    argument_parser.add_argument("--momentum",
                                 default=0.9,
                                 type=float,
                                 help="momentum of SGD")
    
    argument_parser.add_argument("--weight_decay",
                                 default=5e-4,
                                 type=float,
                                 help="weight decay")

    # dataset
    argument_parser.add_argument("--dataset",
                                 default='cifar10',
                                 choices=['cifar10', 'cifar100'])
    
    argument_parser.add_argument('--batch_size',
                                 default=128,
                                 type=int)
    
    argument_parser.add_argument("--drop_last",
                                 default=False,
                                 action='store_true',
                                 help="Whether to drop last mini batch that is smaller than batch_size")
    
    argument_parser.add_argument("--num_workers",
                                 default=0,
                                 type=int)
    
    # load checkpoint
    argument_parser.add_argument("--load_checkpoint",
                                 default=None,
                                 help="checkpoint to load")
    
    args = argument_parser.parse_args()
    
    return args
    

if __name__ == '__main__':

    args = get_args()
    train(args)
