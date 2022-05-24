"""
This code for the MONet model, as well as the weights pretrained on the CLEVR dataset 
are taken and adapted from the repository https://github.com/baudm/MONet-pytorch
"""

from itertools import chain
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations
from utils import *


###############################################################################
# Helper Functions
###############################################################################


class Flatten(nn.Module):

    def forward(self, x):
        return x.flatten(start_dim=1)
    
    
##############################################################################
# MONet Auto-Encoder used on the CLEVR dataset
##############################################################################


class ComponentVAE(nn.Module):

    def __init__(self, input_nc, z_dim=16, full_res=False):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        # full_res = False # full res: 128x128, low res: 64x64
        h_dim = 4096 if full_res else 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc + 1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(h_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 32)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_nc + 1, 1),
        )
        self._bg_logvar = 2 * torch.tensor(0.09).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        output = self.decoder(z_sb)
        x_mu = output[:, :self._input_nc]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self._input_nc:]

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):

    def __init__(self, input_nc, output_nc, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
        self._resize = resize

    def forward(self, *inputs):
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = skip = F.relu(x)
        if self._resize:
            x = F.interpolate(skip, scale_factor=0.5 if downsampling else 2., mode='nearest')
        return (x, skip) if downsampling else x


class Attention(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Attention, self).__init__()
        self.downblock1 = AttentionBlock(input_nc + 1, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.downblock5 = AttentionBlock(ngf * 8, ngf * 8, resize=False)
        # no resizing occurs in the last block of each path
        # self.downblock6 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * ngf * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * ngf * 8),
            nn.ReLU(),
        )

        # self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock2 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock3 = AttentionBlock(2 * ngf * 8, ngf * 4)
        self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock5 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock6 = AttentionBlock(2 * ngf, ngf, resize=False)

        self.output = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, x, log_s_k):
        # Downsampling blocks
        x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        skip6 = skip5
        # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        # _, skip6 = self.downblock6(x)
        # Flatten
        x = skip6.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip6.shape)
        # Upsampling blocks
        # x = self.upblock1(x, skip6)
        x = self.upblock2(x, skip5)
        x = self.upblock3(x, skip4)
        x = self.upblock4(x, skip3)
        x = self.upblock5(x, skip2)
        x = self.upblock6(x, skip1)
        # Output layer
        logits = self.output(x)
        x = F.logsigmoid(logits)
        return x, logits


class MONet(nn.Module):

    def __init__(self, input_nc=3, z_dim=16, num_slots=4):
        """Initialize this model class."""
        super(MONet, self).__init__()
        self.input_nc = input_nc
        self.z_dim = z_dim
        self.num_slots = num_slots
        self.netAttn = Attention(input_nc, 1)
        self.netCVAE = ComponentVAE(input_nc, z_dim)
        try:
            self.netAttn.load_state_dict(torch.load(os.path.join('saved_models', 'latest_net_Attn.pth')))
            self.netCVAE.load_state_dict(torch.load(os.path.join('saved_models', 'latest_net_CVAE.pth')))
        except FileNotFoundError:
            raise FileNotFoundError("We could not find the pretrained weights in the saved_models/ repository. Please make sure you download the pretrained attention network and component VAE of the MONet model by running the download_monet_model.sh script.")
            
        self.code_dim = num_slots * z_dim
        self.input_dim = input_nc*64*64
        self.training = False  # We do not provide the code to train this model
        
    def encode(self, x):
        
        # Initial s_k = 1: shape = (N, 1, H, W)
        shape = list(x.shape)
        shape[1] = 1
        log_s_k = x.new_zeros(shape)

        z_mu = []
        z_logvar = []
        
        for k in range(self.num_slots):
            # Derive mask from current scope
            if k != self.num_slots - 1:
                log_alpha_k, alpha_logits_k = self.netAttn(x, log_s_k)
                log_m_k = log_s_k + log_alpha_k
                # Compute next scope
                log_s_k += -alpha_logits_k + log_alpha_k
            else:
                log_m_k = log_s_k

            # Get z_k parameters
            params = self.netCVAE.encoder(torch.cat((x, log_m_k), dim=1))
            z_mu_k = params[:, :self.netCVAE._z_dim]
            z_logvar_k = params[:, self.netCVAE._z_dim:]
            
            # Save the representation
            z_mu.append(z_mu_k.unsqueeze(1))
            z_logvar.append(z_logvar_k.unsqueeze(1))
            
        return torch.cat(z_mu, dim=1), torch.cat(z_logvar, dim=1)

    def decode(self, z):

        z = z.reshape(-1, self.num_slots, self.z_dim)
        
        m_logits = []
        x_mu = []
        
        for k in range(self.num_slots):

            z_sb = self.netCVAE.spatial_broadcast(z[:, k], 64 + 8, 64 + 8)
            output = self.netCVAE.decoder(z_sb)
            x_mu_k = output[:, :self.netCVAE._input_nc]
            x_mu.append(x_mu_k)
            m_logits_k = output[:, self.netCVAE._input_nc:]
            m_logits.append(m_logits_k)
        
        m_logits = torch.cat(m_logits, axis=1)
        m = m_logits.log_softmax(dim=1).unsqueeze(2).repeat(1, 1, 3, 1, 1).exp()
        x_tilde = 0

        for k in range(self.num_slots):
            x_k_masked = m[:, k] * x_mu[k]
            x_tilde += x_k_masked

        return x_tilde
    
    
##############################################################################
# Convolutional Auto-Encoder used on the CIFAR dataset
##############################################################################


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

    
class CIFARConvEncoder(nn.Module):
    
    """
    Convolutional encoder for the CIFAR10 images
    """
    
    def __init__(self, code_dim, feature_dim):
        super(CIFARConvEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.code_dim = code_dim
        self.conv1 = nn.Conv2d(3, feature_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_dim)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_dim*2)
        self.conv3 = nn.Conv2d(feature_dim*2, feature_dim*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(feature_dim*4)
        self.linear = nn.Linear(feature_dim*4*4*4, 2 * code_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.code_dim]
        logvar = x[:, self.code_dim:]
        return mu, logvar

    
class CIFARConvDecoder(nn.Module):
    
    """
    Convolutional decoder for the CIFAR10 images
    """
    
    def __init__(self, code_dim, feature_dim):
        super(CIFARConvDecoder, self).__init__()
        self.feature_dim = feature_dim*2
        self.code_dim = code_dim
        self.linear = nn.Linear(code_dim, feature_dim*4*4*4)
        self.conv3 = ResizeConv2d(feature_dim*4, feature_dim*2, kernel_size=3, scale_factor=2)
        self.bn3 = nn.BatchNorm2d(feature_dim*2)       
        self.conv2 = ResizeConv2d(feature_dim*2, feature_dim, kernel_size=3, scale_factor=2)
        self.bn2 = nn.BatchNorm2d(feature_dim)        
        self.conv1 = ResizeConv2d(feature_dim, 3, kernel_size=3, scale_factor=2)
        self.feature_dim = feature_dim

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), self.feature_dim*4, 4, 4)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 32, 32)
        return x

    
class CIFARConvAE(nn.Module):
    
    """
    Convolutional Auto-Encoder for the CIFAR10 images
    """
    
    def __init__(self, feature_dim, code_dim):
        super(CIFARConvAE, self).__init__()
        self.feature_dim = feature_dim
        self.code_dim = code_dim
        
        self.encoder = CIFARConvEncoder(code_dim, feature_dim)
        self.decoder = CIFARConvDecoder(code_dim, feature_dim)
        
        try:
            self.load_state_dict(torch.load(os.path.join('saved_models', 'CIFAR10_ae.pth')))
            self.training=False
        except FileNotFoundError:
            self.training=True
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    
##############################################################################
# Benchmark models classes
##############################################################################


class MemN2N(nn.Module):
    
    """
    An implementation of the MemN2N model https://arxiv.org/abs/1503.08895
    """
    
    def __init__(self, mem_dim, num_steps=1):
        
        super(MemN2N, self).__init__()
        self.input_dim = mem_dim
        self.output_dim = mem_dim
        self.mem_dim = mem_dim
        self.key_dim = mem_dim
        self.value_dim = mem_dim
        self.num_steps = num_steps
                        
        self.out_m = nn.Linear(mem_dim, self.key_dim, bias=False)
        self.out_c = nn.Linear(mem_dim, self.key_dim, bias=False)
        
        self.out_u = nn.Linear(self.input_dim, self.key_dim, bias=False)
        self.out_a = nn.Linear(self.key_dim, self.output_dim, bias=False)
        
        self.beta = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.init_weights()
    
    def init_weights(self):
        
        self.out_m.weight.data = torch.eye(self.output_dim)/self.key_dim**.5
        self.out_c.weight.data = torch.eye(self.output_dim)
        self.out_u.weight.data = torch.eye(self.output_dim)/self.key_dim**.5
        self.out_a.weight.data = torch.eye(self.output_dim)
    
    
    def set_mem(self, M):
        
        self.M = M
        
    def forward(self, x, z, alpha=None, return_all=False):
        
        beta = alpha if alpha is not None else self.beta
        
        mem_size = self.M.shape[1]

        u_init = self.out_u(z)
        M = self.M.reshape(-1, self.mem_dim)
        
        u = u_init.unsqueeze(-1)
        
        for step in range(self.num_steps):
        
            m = self.out_m(self.M).reshape(-1, mem_size, self.key_dim)
            c = self.out_c(self.M).reshape(-1, mem_size, self.value_dim)
            
            p = torch.softmax(self.beta * alpha * torch.bmm(m, u), axis=1).transpose(1, 2)
            o = torch.bmm(p, c).transpose(1, 2)
            
            u = u + o                
        
        a = self.out_a(o.squeeze(-1))
        
        if return_all:
            return a, m, c, p, o, u_init
        
        return a


class NTM(nn.Module):
    
    """
    An implementation of a content-based read-only NTM model https://arxiv.org/abs/1410.5401
    The interpolation, shifting and writing mechanisms are not implemented
    """
    
    def __init__(self, mem_dim, read_heads=1):
        
        super(NTM, self).__init__()
        self.mem_dim = mem_dim
        self.read_heads = read_heads
                
        self.out_k = nn.Linear(mem_dim, mem_dim * read_heads, bias=False)
        self.out_g = nn.Linear(mem_dim, read_heads, bias=True)
        
    def set_mem(self, M):
        
        self.M = M
        
    def forward(self, x, z):
        
        mem_size = self.M.shape[1]
                
        # NTM controller
        k = self.out_k(z).reshape(-1, self.read_heads, self.mem_dim)
        g = torch.exp(self.out_g(z)).unsqueeze(1).repeat(1, mem_size, 1)
                
        # Content-based attention
        w = torch.softmax(
            torch.bmm(
                self.M / torch.norm(self.M, dim=-1).unsqueeze(-1).repeat(1, 1, self.mem_dim),
                (k/ torch.norm(k, dim=-1).unsqueeze(-1).repeat(1, 1, self.mem_dim)).transpose(1, 2)
            ),
            axis=1
        )
             
        # Parameterized softmax
        w = torch.softmax(g * torch.log(1e-5 + w), axis=1)
                    
        # Reading
        r = torch.bmm(w.transpose(1, 2), self.M)
        z = torch.mean(r, axis=1)

        return z

def update_hopfield(z, means, beta):
        
    """ 
    Updates z using the update rule of the MCHN model
    """
    
    batch_size, code_dim = z.shape
    N = means.shape[0]

    means = means.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size*N, 1, code_dim).transpose(1, 2)

    z = z.unsqueeze(1).repeat(1, N, 1).reshape(batch_size*N, 1, code_dim).transpose(1, 2)
    
    weights = torch.softmax(beta * torch.bmm(z.transpose(1, 2), means).reshape(batch_size, N), axis=1)
    
    return torch.bmm(weights.unsqueeze(1), means.reshape(batch_size, N, -1)).squeeze(1)


class MCHN(nn.Module):
    
    """ 
    An implementation of the MCHN model 
    """
    
    def __init__(self, mem_dim):
        
        super(MCHN, self).__init__()
        self.mem_dim = mem_dim
        
    def set_mem(self, means):
        
        self.means = means
        
    def forward(self, x, z, beta, iterations=10):
        
        batch_size, N, _ = self.means.shape

        # Reshape
        means = self.means.reshape(batch_size*N, 1, self.mem_dim).transpose(1, 2)
        
        for _ in range(iterations):
            
            # Reshape
            z = z.unsqueeze(1).repeat(1, N, 1).reshape(batch_size*N, 1, self.mem_dim).transpose(1, 2)

            # Attention coefficients
            weights = torch.softmax(beta * torch.bmm(z.transpose(1, 2), means).reshape(batch_size, N), axis=1)
            
            # MCHN update
            z = torch.bmm(weights.unsqueeze(1), means.reshape(batch_size, N, -1)).squeeze(1)
            
        return z

    
##############################################################################
# Our models
##############################################################################


class GMM(nn.Module):
    
    """ 
    An implementation of the GMM model 
    """
    
    def __init__(self, mem_dim):
        
        super(GMM, self).__init__()
        self.mem_dim = mem_dim
        
    def set_mem(self, means):
        
        self.means = means
        
    def forward(self, x, z, sigma, iterations=10):
        
        batch_size, N, _ = self.means.shape
        
        # Reshape
        means = self.means.reshape(batch_size*N, 1, self.mem_dim).transpose(1, 2)

        for _ in range(iterations):
            
            # Reshape
            z = z.unsqueeze(1).repeat(1, N, 1).reshape(batch_size*N, 1, self.mem_dim).transpose(1, 2)
            
            # Attention coefficients
            weights = torch.softmax(-(0.5/sigma**2)*torch.bmm((z-means).transpose(1, 2), z-means).reshape(batch_size, N), axis=1)
            
            # MCHN update
            z = torch.bmm(weights.unsqueeze(1), means.reshape(batch_size, N, -1)).squeeze(1)
                
        return z
    
    
class BPGMM(nn.Module):
    
    """
    An implementation of the BPGMM model
    """
    
    def __init__(self, mem_dim, vae):
        
        super(BPGMM, self).__init__()
        self.mem_dim = mem_dim
        self.vae = vae.eval()
        
    def set_mem(self, means):
        
        self.means = means
        
    def forward(self, x, z, sigma, lr, gamma, iterations=100):
        
        qz_mean = torch.nn.Parameter(z)
        optimizer = torch.optim.Adam([qz_mean], lr=lr)
        
        for _ in range(iterations):
            optimizer.zero_grad()
            x_pred = self.vae.decode(qz_mean)
            loss = nn.MSELoss()(x_pred, x) + gamma * torch.mean(gmm_likelihood(qz_mean, self.means, sigma))
            loss.backward()
            optimizer.step()
            
        return qz_mean.detach()


class GMMStar(nn.Module):
    
    """ 
    An implementation of the GMM model 
    """
    
    def __init__(self, mem_dim):
        
        super(GMMStar, self).__init__()
        self.mem_dim = mem_dim
        self.precision = torch.nn.Parameter(torch.eye(self.mem_dim))
        
    def set_mem(self, means):
        
        self.means = means
        
    def forward(self, x, z, iterations=10):
        
        batch_size, N, _ = self.means.shape
        
        # Reshape
        means = self.means.reshape(batch_size*N, self.mem_dim, 1)

        for _ in range(iterations):
            
            # Reshape
            z = z.unsqueeze(1).repeat(1, N, 1).reshape(batch_size*N, self.mem_dim, 1)

            # Precision
            precision = self.precision.unsqueeze(0).repeat(N*batch_size, 1, 1)
            
            # Attention coefficients
            weights = torch.softmax(
                -0.5 * torch.bmm(
                    (means-z).transpose(1, 2),
                    torch.bmm(
                        precision,
                        means-z
                    )
                ).reshape(batch_size, N),
                axis=1
            )
            
            # MCHN update
            z = torch.bmm(weights.unsqueeze(1), means.reshape(batch_size, N, -1)).squeeze(1)
                
        return z


##############################################################################
# Convolutional Auto-Encoder used on the MNIST dataset
##############################################################################


class MNISTConvEncoder(nn.Module):
    
    """ 
    Convolutional variational encoder for MNIST images
    """
    
    def __init__(self, feature_dim, code_dim):
        super(MNISTConvEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.code_dim = code_dim
        
        self.c1 = nn.Conv2d(1, feature_dim, 4, 2)
        self.c2 = nn.Conv2d(feature_dim, 2*feature_dim, 3, 2)
        self.c3 = nn.Conv2d(2*feature_dim, 4*feature_dim, 4, 2)
        self.mean_head = nn.Linear(2*2*4*feature_dim, code_dim)
        self.logvar_head = nn.Linear(2*2*4*feature_dim, code_dim)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.reshape(batch_size, -1)
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        return mean, logvar
    
    
class MNISTConvDecoder(nn.Module):
    
    """ 
    Convolutional decoder for MNIST images
    """
    
    def __init__(self, feature_dim, code_dim):
        super(MNISTConvDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.code_dim = code_dim
        
        self.tc1 = nn.ConvTranspose2d(feature_dim, 1, 4, 2)
        self.tc2 = nn.ConvTranspose2d(2*feature_dim, feature_dim, 3, 2)
        self.tc3 = nn.ConvTranspose2d(4*feature_dim, 2*feature_dim, 4, 2)
        self.linear = nn.Linear(code_dim, 2*2*4*feature_dim)
        
    def forward(self, z4):
        
        batch_size = z4.shape[0]
    
        z3 = F.relu(self.linear(z4))
        z3 = z3.reshape(batch_size, 4*self.feature_dim, 2, 2)
        z2 = F.relu(self.tc3(z3))
        z1 = F.relu(self.tc2(z2))
        x = torch.sigmoid(self.tc1(z1))
        
        return x


class MNISTConvAE(nn.Module):
    
    """
    Convolutional Auto-Encoder for the MNIST images
    """
    
    def __init__(self, feature_dim, code_dim):
        super(MNISTConvAE, self).__init__()
        self.feature_dim = feature_dim
        self.code_dim = code_dim
        
        self.encoder = MNISTConvEncoder(feature_dim, code_dim)
        self.decoder = MNISTConvDecoder(feature_dim, code_dim)
        
    def encode(self, x):
        return self.encoder(x)[0]
    
    def decoder(self, z):
        return self.decoder(z)
    

##############################################################################
# PC Network on the MNIST dataset
##############################################################################


def update_gmm(z, means, sigma):
        
    """ 
    Updates z using the update rule of the GMM model
    """
    
    batch_size, N, mem_dim = means.shape

    means = means.reshape(batch_size*N, 1, code_dim).transpose(1, 2)  
    
    z = z.unsqueeze(1).repeat(1, N, 1).reshape(batch_size*N, 1, code_dim).transpose(1, 2)
        
    weights = torch.softmax(-(0.5/sigma**2)*torch.bmm((z-means).transpose(1, 2), z-means).reshape(batch_size, N), axis=1)
    
    return torch.bmm(weights.unsqueeze(1), means.reshape(batch_size, N, -1)).squeeze(1)


def complexity(z, means, sigma):
    
    """ 
    Computes the top-down influence of the memory component onto the representation z in the PC network
    """
        
    return z - update_gmm(z, means, sigma)


class PCConvDecoder(nn.Module):
        
    """ 
    PC Network built using the ConvDecoder generative model
    """
    
    def __init__(self, decoder, prior_means, alpha):
        super(PCConvDecoder, self).__init__()
                
        self.feature_dim = decoder.feature_dim
        self.code_dim = decoder.code_dim
        
        # Update rate of the PC network dynamics
        self.alpha = alpha
        
        # Top-down weights reproduce the decoder network
        self.tc1 = decoder.tc1
        self.tc2 = decoder.tc2
        self.tc3 = decoder.tc3
        self.linear = decoder.linear
        
        # Bottom-up weights correspond to transposed versions of the top-down weights
        self.c1 = nn.Conv2d(self.feature_dim, 1, 4, 2, bias=False)
        self.c1.weight.data = decoder.tc1.weight.data
        self.c2 = nn.Conv2d(2*self.feature_dim, 2*self.feature_dim, 3, 2, bias=False)
        self.c2.weight.data = decoder.tc2.weight.data
        self.c3 = nn.Conv2d(4*self.feature_dim, 2*self.feature_dim, 4, 2, bias=False)
        self.c3.weight.data = decoder.tc3.weight.data        
        self.tlinear = nn.Linear(2*2*4*self.feature_dim, self.code_dim, bias=False)
        self.tlinear.weight.data = self.linear.weight.data.T
                
    def set_mem(self, means):
        
        self.means = means

        
    def forward(self, z4, x_target, iterations=3, update_predictions=True, save_errors=False, sigma=0.1):
                
        """ 
        PC Network built using the ConvDecoder generative model
        Parameters:
            - z4: initial estimate of the top-most representation, typically provided by the VAE encoder
            - x_target: the observed input
            - iterations: the number of inference iterations
            - update_predictions: whether we use the 'fixed predictions' assumption or not
            - save_errors: whether we save the prediction errors for further visualisation
            - sigma: standard deviation of the prior distribution on z4
        """
        
        batch_size = x_target.shape[0]
        
        # Initialize latent variables
        z4_pred = z4.clone()
        z4 = z4.clone()
        z3 = F.relu(self.linear(z4))
        z3 = z3.reshape(batch_size, 4*self.feature_dim, 2, 2)
        z2 = F.relu(self.tc3(z3))
        z1 = F.relu(self.tc2(z2))
        
        output_prediction_errors = torch.zeros(iterations, batch_size)
        vfe = torch.zeros(iterations, 5, batch_size).cuda()
        
        # Preactivations
        z3_pre = self.linear(z4).reshape(batch_size, 4*self.feature_dim, 2, 2)
        z2_pre = self.tc3(z3)
        z1_pre = self.tc2(z2)
        x_pre = self.tc1(z1)
        
        # Save prediction errors
        if save_errors:
            self.eps_4s = torch.zeros(iterations, batch_size, self.code_dim).cpu()
            self.eps_3s = torch.zeros(iterations, batch_size, 4*self.feature_dim, 2, 2).cpu()
            self.eps_2s = torch.zeros(iterations, batch_size, 2*self.feature_dim, 6, 6).cpu()
            self.eps_1s = torch.zeros(iterations, batch_size, self.feature_dim, 13, 13).cpu()
            
        for i in range(iterations):
            
            # Preactivations
            if update_predictions:
                z3_pre = self.linear(z4).reshape(batch_size, 4*self.feature_dim, 2, 2)
                z2_pre = self.tc3(z3)
                z1_pre = self.tc2(z2)
                x_pre = self.tc1(z1)
            
            # Compute prediction errors at each layer
            if update_predictions:
                if self.means is None:
                    eps_z4 = torch.zeros_like(z4)
                else:
                    eps_z4 = complexity(z4, self.means, sigma)            
            else:
                eps_z4 = z4 - z4_pred

            eps_z3 = z3 - F.relu(z3_pre)
            eps_z2 = z2 - F.relu(z2_pre)
            eps_z1 = z1 - F.relu(z1_pre)
            eps_x = x_target - torch.sigmoid(x_pre)
            
            # Update representations
            z4 += self.alpha * (self.tlinear((threshold(z3_pre) * eps_z3).reshape(batch_size, self.feature_dim*4*2*2)) - eps_z4)
            z3 += self.alpha * (self.c3(threshold(z2_pre) * eps_z2) - eps_z3)
            z2 += self.alpha * (self.c2(threshold(z1_pre) * eps_z1) - eps_z2)
            z1 += self.alpha * (self.c1(derivative_sigmoid(x_pre) * eps_x) - eps_z1)

            # Compute free energy
            vfe[i, 0] = torch.sum(eps_x.reshape(batch_size, -1)**2, axis=-1)
            vfe[i, 1] = torch.sum(eps_z1.reshape(batch_size, -1)**2, axis=-1)
            vfe[i, 2] = torch.sum(eps_z2.reshape(batch_size, -1)**2, axis=-1)
            vfe[i, 3] = torch.sum(eps_z3.reshape(batch_size, -1)**2, axis=-1)
            vfe[i, 4] = torch.sum(eps_z4.reshape(batch_size, -1)**2, axis=-1)
            
            prediction_error = torch.mean(eps_x.reshape(batch_size, -1)**2, axis=1)
            output_prediction_errors[i] = prediction_error

            if save_errors:
                self.eps_4s[i] = eps_z4.detach().cpu()
                self.eps_3s[i] = eps_z3.detach().cpu()
                self.eps_2s[i] = eps_z2.detach().cpu()
                self.eps_1s[i] = eps_z1.detach().cpu()           
            
        return torch.sigmoid(x_pre), z1, z2, z3, z4, output_prediction_errors, vfe
