import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.image_pool import ImagePool
from .base_model import BaseModel
import itertools
from models import networks
from config import cfg


class SFABlock3(nn.Module):
    """Split Frequency Attention Block"""
    def __init__(self, in_channels, out_channels):
        super(SFABlock3, self).__init__()
        self.cos_branch_conv1 = nn.Conv2d(
            in_channels, out_channels, 3, 1, 1
        )
        self.cos_branch_conv1_norm = nn.InstanceNorm2d(in_channels)
        self.sin_branch_conv1 = nn.Conv2d(
            in_channels, out_channels, 3, 1, 1
        )
        self.sin_branch_conv1_norm = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        x_complex = torch.fft.fft2(x, norm='backward')
        # split
        x_cos = x_complex.real
        x_sin = x_complex.imag
        # branch forward
        x_cos = self.cos_branch_conv1(x_cos)
        x_cos = self.cos_branch_conv1_norm(x_cos)
        x_sin = self.sin_branch_conv1(x_sin)
        x_sin = self.sin_branch_conv1_norm(x_sin)
        # concat
        x_complex = (x_cos).type(torch.complex64) + (1j*x_sin).type(torch.complex64)
        out = torch.fft.ifft2(x_complex,  norm='backward').real
        out = F.relu(out)
        return out


class SFABlock2(nn.Module):
    """Split Frequency Attention Block. Refer of Fourier Unit (FU)"""
    def __init__(self, ngf):
        super(SFABlock2, self).__init__()
        self.ngf = ngf
        self.deconv1 = nn.ConvTranspose2d(
            ngf * 8, ngf * 4, 3, 2, 1, 1
        )
        self.deconv1_norm = nn.InstanceNorm2d(ngf * 2)

    def forward(self, x):
        x_complex = torch.fft.fft2(x, norm='backward')
        # split
        x_r = x_complex.real
        x_i = x_complex.imag
        x_cat = torch.cat([x_r, x_i], dim=1)
        # convolution forward
        x_cat = self.deconv1(x_cat)
        x_cat = self.deconv1_norm(x_cat)
        x_cat = F.relu(x_cat)

        # split
        x_r, x_i = torch.split(x_cat, split_size_or_sections=self.ngf * 2, dim=1)

        # concat
        x_complex = (x_r).type(torch.complex64) + (1j*x_i).type(torch.complex64)
        out = torch.fft.ifft2(x_complex,  norm='backward').real
        return out


class SFABlock(nn.Module):
    """Split Frequency Attention Block"""
    def __init__(self, ngf):
        super(SFABlock, self).__init__()
        self.cos_branch_deconv1 = nn.ConvTranspose2d(
            ngf * 4, ngf * 2, 3, 2, 1, 1
        )
        self.cos_branch_deconv1_norm = nn.InstanceNorm2d(ngf * 2)
        self.sin_branch_deconv1 = nn.ConvTranspose2d(
            ngf * 4, ngf * 2, 3, 2, 1, 1
        )
        self.sin_branch_deconv1_norm = nn.InstanceNorm2d(ngf * 2)

    def forward(self, x):
        x_complex = torch.fft.fft2(x, norm='backward')
        # split
        x_cos = x_complex.real
        x_sin = x_complex.imag
        # branch forward
        x_cos = self.cos_branch_deconv1(x_cos)
        x_cos = self.cos_branch_deconv1_norm(x_cos)
        x_cos = F.relu(x_cos)
        x_sin = self.sin_branch_deconv1(x_sin)
        x_sin = self.sin_branch_deconv1_norm(x_sin)
        x_sin = F.relu(x_sin)
        # concat
        x_complex = (x_cos).type(torch.complex64) + (1j*x_sin).type(torch.complex64)
        out = torch.fft.ifft2(x_complex,  norm='backward')
        out = F.relu(out)
        return out


class FCVGANmodel(BaseModel):

    def __init__(self, isTrain=False):
        """Initialize the CycleGAN class.
        """
        BaseModel.__init__(self, isTrain)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'o1_b', 'o2_b', 'o3_b', 'o4_b', 'o5_b', 'o6_b', 'o7_b', 'o8_b', 'o9_b', 'o10_b',
        'a1_b', 'a2_b', 'a3_b', 'a4_b', 'a5_b', 'a6_b', 'a7_b', 'a8_b', 'a9_b', 'a10_b', 'i1_b', 'i2_b', 'i3_b', 'i4_b', 'i5_b',
        'i6_b', 'i7_b', 'i8_b', 'i9_b']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'o1_a', 'o2_a', 'o3_a', 'o4_a', 'o5_a', 'o6_a', 'o7_a', 'o8_a', 'o9_a', 'o10_a',
        'a1_a', 'a2_a', 'a3_a', 'a4_a', 'a5_a', 'a6_a', 'a7_a', 'a8_a', 'a9_a', 'a10_a', 'i1_a', 'i2_a', 'i3_a', 'i4_a', 'i5_a',
        'i6_a', 'i7_a', 'i8_a', 'i9_a']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'a10_b', 'real_B','fake_A', 'a10_a']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B


        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(3, 3, cfg.ngf, cfg.netG, cfg.norm,
                                        not cfg.no_dropout, cfg.init_type, cfg.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(3, 3, cfg.ngf, cfg.netG, cfg.norm,
                                        not cfg.no_dropout, cfg.init_type, cfg.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(3, cfg.ndf, cfg.netD, 3,
                                            cfg.norm, cfg.init_type, cfg.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(3, cfg.ndf, cfg.netD, 3,
                                            cfg.norm, cfg.init_type, cfg.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(cfg.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(cfg.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(cfg.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # self.real_A = input['A'].to(self.device)
        # self.real_B = input['B'].to(self.device)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
            self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
            self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(self.real_A)  # G_A(A)

        self.rec_A, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _ = self.netG_B(self.fake_B)   # G_B(G_A(A))

        self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
            self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
            self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(self.real_B)  # G_B(B)

        self.rec_B, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _ = self.netG_A(self.fake_A)   # G_A(G_B(B))



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = cfg.lambda_identity
        lambda_A = cfg.lambda_A
        lambda_B = cfg.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _  = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _  = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
