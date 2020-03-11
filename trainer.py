# trainer that implements base supervised learning algorithm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch.utils.tensorboard import SummaryWriter

import os


class SupervisedTrainer:
    def __init__(self, model, train_dataloader, validation_dataloader, n_class=10,  # model and data
                 label_smoothing=True, alpha=0.1, learning_rate=1e-1, momentum=0.9, weight_decay=5e-4,  # training hyperparameter
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        self.device = device

        self.model = model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[32000, 48000])
        
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.n_class = n_class
        
        if label_smoothing:
            self.criterion = LabelSmoothingCrossEntropyLoss(alpha=alpha, n_class=n_class)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.curr_epoch = 0
        self.curr_step = 0
        
        self.best_val_acc = 0
        
        self.writer = SummaryWriter()

    def train(self, n_epochs):
        # training process
        
        for epoch in range(self.curr_epoch, n_epochs):
            self.curr_epoch += 1
            
            epoch_loss = 0
            epoch_correct = 0
            epoch_count = 0
            start_time = time.time()
            for ix, (x, y) in enumerate(self.train_dataloader):
                self.curr_step += 1

                # run train step
                x, y = x.to(self.device), y.to(self.device)
                
                loss, correct = self.train_step(x, y)
                self.writer.add_scalar("Train/MinibatchLoss", loss, self.curr_step)
                epoch_loss += loss
                epoch_correct += correct
                epoch_count += len(x)
                
                # lr scheduler update
                self.scheduler.step()
                
            train_loss = epoch_loss / (ix + 1)
            train_acc = epoch_correct / epoch_count

            # print result of an epoch
            print("Epoch [{0}/{1}] Loss: {2:.4f}, Accuracy: {3:.4f}({4}/{5}) time: {6:.2f}s".format(self.curr_epoch, n_epochs, 
                                                                         train_loss, 
                                                                         train_acc,
                                                                         epoch_correct,
                                                                         epoch_count,
                                                                         time.time() - start_time))
            
            self.writer.add_scalar("Train/Loss", train_loss, self.curr_epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, self.curr_epoch)
                
            val_loss, val_acc = self.validate()
            print("Validation: Epoch [{0}/{1}] Validation Loss: {2:.4f}, Validation Accuracy: {3:.4f}".format(self.curr_epoch, n_epochs, 
                                                                                                    val_loss,
                                                                                                    val_acc))
            
            self.writer.add_scalar("Validation/Loss", val_loss, self.curr_epoch)
            self.writer.add_scalar("Validation/Accuracy", val_acc, self.curr_epoch)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

                if not os.path.isdir("temp_checkpoints"):
                    os.makedirs("temp_checkpoints")
                
                self.save_checkpoint("temp_checkpoints/best_val_acc.pth")
                
        
    def train_step(self, x, y):
        # train step using supervised method

        self.model.train()
                
        pred = self.model(x)
        loss = self.criterion(pred, y)
        
        correct = torch.sum(pred.argmax(dim=-1) == y).item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        return loss.item(), correct
            
    def validate(self):
        # validation step to calculate validation loss and validation accuracy
        val_loss_sum = 0
        val_correct = 0
        val_count = 0
        self.model.eval()
        with torch.no_grad():
            for ix, (val_x, val_y) in enumerate(self.validation_dataloader):
                val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                pred_logits = self.model(val_x)
                loss = self.criterion(pred_logits, val_y)
                val_loss_sum += loss.item() * len(val_x)
                val_correct += torch.sum(pred_logits.argmax(dim=-1) == val_y).item()
                val_count += len(val_x)
                
        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count
                
        return val_loss, val_acc
    
    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.curr_epoch,
            'step': self.curr_step
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.curr_epoch = checkpoint['epoch']
        self.curr_step = checkpoint['step']
        
        
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, n_class=10):
        super().__init__()
        self.alpha = alpha
        self.n_class = n_class

    def forward(self, logits, target):
        with torch.no_grad():
            one_hot_target = torch.zeros(size=(target.size(0), self.n_class), device=target.device).scatter_(1, target.view(-1, 1), 1)
            uniform_target = torch.ones(size=(target.size(0), self.n_class), device=target.device) * (1 / self.n_class)
            label_smoothed_target = one_hot_target * (1 - self.alpha) + uniform_target * self.alpha

        logprobs = F.log_softmax(logits, dim=-1)
        loss = - (label_smoothed_target * logprobs).sum(-1)
        return loss.mean()
    
