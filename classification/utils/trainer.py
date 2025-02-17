import torch
import torch.nn as nn
import tqdm
from utils.result_utils import *
import wandb
import numpy as np
import random
random.seed(1)

class Trainer:

    def __init__(self, model, data_train, data_valid, data_test, args, device):
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.args = args
        self.device = device

    def train(self):
        self.model = self.model.to(self.device)
        loss_fn=nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.train.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.train.epoch)

        Epoch_num = self.args.train.epoch

        for epoch in range(Epoch_num):
            print(f"Epoch {epoch + 1}")

            idx = 0
            for x_batch, y_batch in tqdm.tqdm(self.data_train):
                idx+=1

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)    
                y_batch_pred = self.model(x_batch)
                loss = loss_fn(y_batch_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_acc, valid_loss = test(self.data_valid, self.device, self.model, loss_fn, 'valid', self.args)
            print('validation accuracy: ', valid_acc, 'validation loss: ', valid_loss) 

            test_acc, test_loss = test(self.data_test, self.device, self.model, loss_fn, 'test', self.args)
            print('test accuracy: ', test_acc, 'test loss: ', test_loss) 

            wandb.log({
                    'valid_loss': valid_loss, 
                    "valid_acc": valid_acc,
                    'test_loss': test_loss,
                    "test_acc": test_acc})

            lr_scheduler.step()
        