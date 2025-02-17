from torch.utils.data import DataLoader
from utils.transform_utils import *
from utils.dataset import RadarDataset
from torchvision import transforms
import pandas as pd
import random

random.seed(1)

def LoadDataset(args):
    """Do transforms on radar data and labels. Load the data.

    Args:
        args: args configured in Hydra YAML file

    """
    df = pd.read_csv(args.result.file_list)
   
    df = df.loc[ (df['set'] != args.result.exp_set )]
    subjects = list(set(df['subject']))

    # specifying test subjects
    test_subjects = args.result.test_subjects
    train_subjects = list(set(subjects) - set(test_subjects))

    file_list_train = df.loc[ (df['subject'].isin(train_subjects)) ]
    file_list_train = list(file_list_train['fname'])
    file_list_test = df.loc[ (df['subject'].isin(test_subjects)) & (df['test_subject'] == 1 )]
    file_list_test = list(file_list_test['fname'])
    file_list_valid = df.loc[ (df['subject'].isin(test_subjects)) & (df['test_subject'] == 2 )]
    file_list_valid = list(file_list_valid['fname'])
    
    labelmap = LabelMap(label_type=args.transforms.label_type, ymap=args.transforms.ymap_pattern)
    composed_label = transforms.Compose([labelmap, ToOneHot(args.train.num_classes)])
    
    ### Compose the transforms on train set
    radar_train_dataset_ls = []
    radar_valid_dataset_ls = []
    radar_test_dataset_ls = []
    for radar_idx in args.transforms.select_radar_idx:
        radar_train = RadarDataset(
                            file_list=file_list_train,
                            transform = CropA(),
                            target_transform=composed_label,
                            label_type=args.transforms.label_type,
                            return_des=args.result.return_des_train,
                            is_sim=args.train.train_is_sim,
                            real_data_dir=args.result.real_data_dir,
                            inference_data_dir=args.result.inference_data_dir,
                            is_coarse = args.train.train_is_coarse,
                            radar_idx = radar_idx,
                            )
        radar_train_dataset_ls.append(radar_train)

        for crop_x in args.transforms.crop_x_ls:
            radar_valid = RadarDataset(
                                        file_list=file_list_valid,
                                        transform=CropA_defined(crop_x), 
                                        target_transform=composed_label,
                                        label_type=args.transforms.label_type,
                                        is_sim=args.train.valid_is_sim,
                                        real_data_dir=args.result.real_data_dir,
                                        inference_data_dir=args.result.inference_data_dir,
                                        return_des=args.result.return_des_valid,
                                        is_coarse = args.train.valid_is_coarse,
                                        radar_idx = radar_idx,
                                        )

            radar_test =  RadarDataset(
                                        file_list=file_list_test,
                                        transform =CropA_defined(crop_x),
                                        target_transform = composed_label,
                                        label_type=args.transforms.label_type,
                                        return_des=args.result.return_des_test,
                                        is_sim=args.train.test_is_sim,
                                        real_data_dir=args.result.real_data_dir,
                                        inference_data_dir=args.result.inference_data_dir,
                                        is_coarse = args.train.test_is_coarse,
                                        radar_idx = radar_idx,
                                        )
            
            
            radar_valid_dataset_ls.append(radar_valid)
            radar_test_dataset_ls.append(radar_test)


    ## Concatenate one-channel data from two radars
    radar_dataset_train = torch.utils.data.ConcatDataset(radar_train_dataset_ls)
    radar_dataset_valid = torch.utils.data.ConcatDataset(radar_valid_dataset_ls)
    radar_dataset_test = torch.utils.data.ConcatDataset(radar_test_dataset_ls)
    
    data_train = DataLoader(radar_dataset_train, batch_size=args.train.batch_size, shuffle=args.train.shuffle, num_workers=args.train.num_workers)
    data_valid = DataLoader(radar_dataset_valid, batch_size=args.train.batch_size, shuffle=args.train.shuffle, num_workers=args.train.num_workers)
    data_test = DataLoader(radar_dataset_test, batch_size=args.train.batch_size, shuffle=args.train.shuffle, num_workers=args.train.num_workers)

    return data_train, data_valid, data_test


