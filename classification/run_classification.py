import torch
import sys
import os
from utils.dataloader import LoadDataset
from utils.result_utils import save_result
from utils.model import MyMobileNet
from utils.trainer import Trainer
import hydra
from omegaconf.dictconfig import DictConfig
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1)

@hydra.main(version_base="1.2", config_path="conf", config_name="config_classification")
def main(args: DictConfig) -> None:
    # Logging the configs
    config = {"epoch": int(args.train.epoch),
            "learning_rate": float(args.train.learning_rate),
            "name": args.result.name,
            "win_size": int(args.transforms.win_size),
            }
    # Initialize wandb project
    wandb.init(
        project = args.wandb.project, 
        config = config, 
        notes = args.wandb.notes,
        name = args.result.name,
        )
    device = torch.device('cuda:'+str(args.result.gpu) if torch.cuda.is_available() else 'cpu')
    data_train, data_valid, data_test = LoadDataset(args)

    model = MyMobileNet(num_classes=args.train.num_classes)
    
    ### Training
    trainer = Trainer(model=model, 
                    data_train=data_train, 
                    data_valid=data_valid, 
                    data_test=data_test, 
                    args=args, 
                    device=device,
                    )
    trainer.train()

    ### Saving results for validation and test sets
    print('saving validation and test results---')
    sets = (data_test, data_valid)    
    set_names = ('test', 'valid')  
    for result_set, set_name in zip(sets, set_names): 
        save_result(
            set=result_set,
            set_name=set_name,
            device=device,
            model=model,
            args=args,
            )
        
    wandb.finish()

if __name__ == "__main__":
    main()