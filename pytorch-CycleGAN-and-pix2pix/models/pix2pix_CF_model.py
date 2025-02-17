import torch
from .base_model import BaseModel
from . import networks
from .model_classifier import MyMobileNet
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch.nn.functional as F
import numpy as np

# Classifier Loss
class ClassifierNet(nn.Module):
    def __init__(self, device, opt):
        super(ClassifierNet, self).__init__()
        self.ClassifierNet = MyMobileNet()
        self.ClassifierNet.load_state_dict(torch.load(opt.classifier_path, map_location=device))
        self.ClassifierNet.eval()
        self.ClassifierNet.to(device)
        

    def forward(self, x):
        x = self.ClassifierNet(x)
        x = torch.softmax(x, dim=1)
        return x


class pix2pixCFmodel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet128' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

            parser.add_argument('--lambda_classifier', type=float, default=45.0, help='weight for classifier loss')
            parser.add_argument('--lambda_feature', type=float, default=50.0, help='weight for feature loss')
            parser.add_argument('--Closs_start_epoch', type=int, default=200, help='start epoch for classifier loss')
            parser.add_argument('--Floss_start_epoch', type=int, default=20, help='start epoch for feature loss')
            parser.add_argument('--noise_level', type=float, default=0.08, help='noise level')
            parser.add_argument('--classifier_path', type=str, default='./pretrained_classifier/pretrained_classifier_set1.pt', help='path to pretrained model')
            parser.add_argument('--slowD_start_epoch', type=int, default=100, help='start epoch for slowly updating D')
            parser.add_argument('--slowD_step', type=int, default=5, help='step for slowly updating D')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_classifier', 'G_feature', 'D_fake_acc', 'D_real_acc']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.classifier = ClassifierNet(self.device, opt)
        train_nodes, eval_nodes = get_graph_node_names(ClassifierNet(self.device, opt))

        self.return_nodes = {
        'ClassifierNet.MobileNet.features.3': 'layer3',
        'ClassifierNet.MobileNet.features.6': 'layer6',
        'ClassifierNet.MobileNet.features.15': 'layer15',
        'ClassifierNet.MobileNet.features.17': 'layer17',
        }
        self.body = create_feature_extractor(self.classifier, return_nodes=self.return_nodes)


        if self.isTrain:  
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionClassifier = torch.nn.CrossEntropyLoss() 
            self.criterionfeature = torch.nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.label = input['label'].to(self.device)
        self.num_classes = 2

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  

    def backward_D(self, noise):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator

        fake_AB = fake_AB + torch.randn(fake_AB.shape).to(self.device) * noise 
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_fake_pred = torch.sigmoid(pred_fake)
        self.loss_D_fake_acc = torch.sum(pred_fake_pred < 0.5).float() / (pred_fake_pred.shape[0] * pred_fake_pred.shape[2] * pred_fake_pred.shape[3])

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)        
        real_AB = real_AB + torch.randn(real_AB.shape).to(self.device) * noise
        
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        pred_real_pred = torch.sigmoid(pred_real)
        self.loss_D_real_acc = torch.sum(pred_real_pred > 0.5).float() / (pred_fake_pred.shape[0] * pred_fake_pred.shape[2] * pred_fake_pred.shape[3])
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self,epoch):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        if epoch > self.opt.Closs_start_epoch:
            with torch.no_grad():
                pred_A = self.netG(self.real_A)
                pred_A = torch.repeat_interleave(pred_A, repeats=3, dim=1)
                real_B = torch.repeat_interleave(self.real_B, repeats=3, dim=1)
                classA = self.classifier(pred_A)
                classB = F.one_hot(torch.tensor(self.label, dtype=torch.int64), self.num_classes)
                classB = classB.type(torch.FloatTensor).to(self.device)
                self.loss_G_classifier = self.criterionClassifier(classA, classB) * self.opt.lambda_classifier 
        else:
            self.loss_G_classifier = 0

         # Feature loss
        if epoch > self.opt.Floss_start_epoch:        
            with torch.no_grad():
                pred_A = self.netG(self.real_A)
                pred_A = torch.repeat_interleave(pred_A, repeats=3, dim=1)
                real_B = torch.repeat_interleave(self.real_B, repeats=3, dim=1)
                self.feature_A = self.body(pred_A)
                self.feature_B = self.body(real_B)
                self.loss_G_feature = 0
                self.A_features_shape = []
                self.A_features = []
                for key in self.return_nodes:
                    self.loss_G_feature += self.criterionfeature(self.feature_A[self.return_nodes[key]], self.feature_B[self.return_nodes[key]])
                    self.A_features_shape.append(self.feature_A[self.return_nodes[key]].shape)
                    self.A_features.append(self.loss_G_feature)
                self.loss_G_feature = self.loss_G_feature/4 * self.opt.lambda_feature

        else:
            self.loss_G_feature = 0
        
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_classifier + self.loss_G_feature
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()                   # compute fake images: G(A)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
            
        self.backward_D(self.opt.noise_level)                # calculate gradients for D
        if  (epoch-1)%self.opt.slowD_step==0 or epoch<self.opt.slowD_start_epoch:
            self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(epoch)                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

