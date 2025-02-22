# High-Resolution Gait Micro-Doppler Synthesis from Videos Over Diverse Trajectories
This is the research repository for High-Resolution Gait Micro-Doppler Synthesis from Videos Over Diverse Trajectories (ICASSP 2025). Here contains the code for
* The data used to train the conditional GAN network. 
* The conditional GAN training and inference code. 
* The evaluation classification training and inference.

## Getting Started
After cloning the git repository, in the folder:
1. Environment setup
```
conda create -n high_res_uD_syn python=3.12
conda activate high_res_uD_syn
pip install -r requirements.txt
```

2. Prepare dataset and metadata
* Download and unzip the dataset (input_output_pairs) from Google Drive <a href="https://drive.google.com/drive/folders/1-WcWbp9jiQaMMYGYxy7UYuBqHI5T65HD?usp=sharing"> here </a>. 
The `input_output_pairs` contain coarse micro-Doppler and real micro-Doppler pairs needed to train the conditional GAN. The data from MVDoppler is split into three sets for cross-subject validation, with each set having 4 subjects. One set will be used to train the conditional GAN (train folder under each set pairs) and two other sets will be used for evaluating the simulator, which is training and testing a classifier. 

* Download and unzip the folder `pretrained_classifier` and put under this repo's `pytorch-CycleGAN-and-pix2pix` folder. Take set 1 as an example, `pretrained_classifier_set1.pt` is pre-trained using set 1 data. This pretrained classifier will be used when set 1 data is used train the conditional GAN.
The code structure will be 
```
  ├── pytorch-CycleGAN-and-pix2pix
        ├── pretrained_classifier
            ├── pretrained_classifier_set1.pt
            ├── ...
        ├── ...
  ├── classification
  ...
```

* Download `snapshot_set_split.csv` and put into this repository (`High-Resolution-Gait-Micro-Doppler-Synthesis-from-Videos-Over-Diverse-Trajectories/snapshot_set_split.csv`). 
This file is generated from MVDoppler dataset, containing metadata, e.g., file name, descriptions, set information.

- exp_fname: episode name 
- fname: snapshot name
- length: signature of length 256 Doppler bins (1.28 seconds)
- pattern: activity class, e.g., normal walking and texting
- sex: subject sex
- age: subject age
- height: subject height
- set: set number. The data from MVDoppler is splitted into three sets for cross-subject validation, with each set having 4 subjects.
- test_subject: Flag that if this snapshot is used for testing the simulator. `1` means it is in test set, and `2` means it is in validation set. `0` means this snapshot is an augmented copy to keep the training data balanced, thus not used in test or validation.

* Download MVDoppler dataset. This will be used in training and testing the evaluation classifier (e.g., training on synthetic data and testing on real data). The directory to MVDoppler dataset needs to specified in `real_data_dir` in `classification/conf/config_classification.yaml`.

## Training and Testing the conditional GAN
1. Optional: for visualizing the training process. Open a visdom server through a defined port (e.g., port 8010). This step needs to be before the training process. 
```
python -m visdom.server -port 8010
```

2. Training the conditional GAN. If selecting set 1 to train the network:
```
cd pytorch-CycleGAN-and-pix2pix

python train.py --dataroot <your data directory for input_output_pairs of set 1> --name set1_CF_network --model pix2pix_CF --gpu_ids 0 --checkpoints_dir <your directory to save the checkpoints> --no_flip --display_port 8010 --classifier_path ./pretrained_classifier/pretrained_classifier_set1.pt
```
Notes:
* The set can be selected from (1,2,3)
* The network name can be customized, e.g., set1_CF_network
* The network can be selected from `pix2pix_GAN` (L1 loss + GAN loss), `pix2pix_C` (L1 loss + GAN loss + classifier loss), `pix2pix_CF` (L1 loss + GAN loss + classifier loss + feature loss)

3. Inference to generate synthetic micro-Doppler
```
python test.py --dataroot <your data directory for input_output_pairs of set 1> --name set1_CF_network --model pix2pix_CF --use_wandb --gpu_ids 0 --results_dir <your directory to save the inferred synthetic data> --num_test 25000 --checkpoints_dir <your directory to save the checkpoints> --classifier_path ./pretrained_classifier/pretrained_classifier_set1.pt --preprocess none --no_dropout --no_flip --eval
```
Then the generated synthetic micro-Doppler signatures are saved in the `results_dir`.

## Training and Testing the Evaluation Classifier
Training and testing the classifier to evaluate the generated synthetic micro-Doppler
```
cd classification
python run_classification.py
```
The configurations are in `classification/conf/conf_classification.yaml`. You need to pecifiy the directories in the .yaml file, e.g., the directory saving all the inferred data.

## Citation
This paper is accepted by ICASSP 2025, If you use this code for your research, please cite our paper (later will be updated by the citation of published version)
```
@article{yang2025high-resolution,
  title={High-Resolution Gait Micro-Doppler Synthesis from Videos Over Diverse Trajectories},
  author={Shubo Yang, Soheil Hor, Jae-Ho Choi, Amin Arbabian},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgments
This project heavily uses code from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git) and data from [MVDoppler](https://mvdoppler.github.io/)
