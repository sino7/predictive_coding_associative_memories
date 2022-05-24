import torch

##############################################################################
# Image transformation functions
##############################################################################


def apply_mask(image_batch, size=8):
    
    """
    Applies a gray random mask of width = 2*size
    """
        
    batch_size, channels, width, _ = image_batch.shape
    mask = torch.ones_like(image_batch)
    x = size + torch.randint(width-size*2, size=(batch_size,))
    y = size + torch.randint(width-size*2, size=(batch_size,))
    for i in range(batch_size):
        mask[i, :, x[i]-size:x[i]+size, y[i]-size: y[i]+size] = 0
    
    result = mask * image_batch + 0.5*(1 - mask)
    
    return result

def apply_noise(image_batch, sigma=0.1):
    
    """
    Applies gaussian noise of variance sigma
    """
    
    res = image_batch.clone() + sigma * torch.randn_like(image_batch, device=image_batch.get_device())
    res = torch.min(res, torch.ones_like(res))
    res = torch.max(res, torch.zeros_like(res))
    return res

def color_rotation(image_batch):
    
    """
    Rotates the RGB channels of the images
    """
    
    result = torch.zeros_like(image_batch, device=image_batch.get_device())
    result[:, 0] = image_batch[:, 1]
    result[:, 1] = image_batch[:, 2]
    result[:, 2] = image_batch[:, 0]
    
    return image_batch

def renormalize(img):
    
    """
    Renormalizes the images of the CLEVR dataset
    """
    
    return 0.5 * img + 0.5


def exp_transform(img, batch_size, exp_name, exp_param=None):
    
    # Reshape input image
    n = torch.numel(img)
    if n==batch_size*3*64*76:
        img = img.reshape(batch_size, 3, 64, 76)
        # Renormalize CLEVR image
        img = renormalize(img)
    elif n==batch_size*3*32*32:
        img = img.reshape(batch_size, 3, 32, 32)
    elif n==batch_size*1*28*28:
        img = img.reshape(batch_size, 1, 28, 28)
    _, channels, height, _ = img.shape
    
    # Apply transformation
    if exp_name == 'shift':
        img = img[:, :, :, -height:]
    else:
        img = img[:, :, :, :height]
        if exp_name == 'clean':
            pass
        elif exp_name == 'color':
            img = color_rotation(img)
        elif exp_name == 'noise':
            img = apply_noise(img, exp_param)
        elif exp_name == 'mask':
            img = apply_mask(img, exp_param//2)
        else:
            print("Unkwown experiment name '" + exp_name + "'.")
    
    # Normalization of the CLEVR dataset
    if img.shape[-1]==64:
        img = 2*(img-0.5)
    
    return img


##############################################################################
# Helper functions
##############################################################################


def threshold(x):
    
    """
    Derivative of the ReLU activation function
    """
    
    return (x>0).float()

def derivative_sigmoid(x):
    
    """
    Derivative of the sigmoid activation function
    """
    
    return torch.sigmoid(x) * (1-torch.sigmoid(x))

def ones(x):
    
    """
    Derivative of the Identity activation function
    """
    
    return 1
    
    
def hopfield_energy(x, means, beta, k=0):
    
    """ 
    Computes the energy function of the MCHN https://arxiv.org/abs/2008.02217
    """
    
    batch_size = x.shape[0]           
    
    batch_size, code_dim = x.shape
    n, _ = means.shape
    
    x = x.unsqueeze(2)
    means = means.unsqueeze(0).repeat(batch_size, 1, 1)
    
    lse = -(1/beta) * torch.logsumexp(
        beta * torch.bmm(means, x).reshape(batch_size, -1),
        axis=1
    )
    
    norm = .5 * torch.sum(x.squeeze(-1)**2, axis=1)
        
    return lse + norm


def gmm_likelihood(x, means, sigma):
    
    """ 
    Computes the VFE for the GMM model (up to additive and multiplicative constants)
    """
    
    batch_size, N, mem_dim = means.shape
    
    x = x.unsqueeze(1).repeat(1, N, 1)
    
    lse = - torch.logsumexp(
        -(.5/sigma**2) * torch.sum((means - x)**2, axis=2),
        axis=1
    )
    
    return lse