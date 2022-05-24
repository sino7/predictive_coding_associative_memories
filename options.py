from utils import *
from models import *
from datasets import *


class VAEOptions():
    
    def __init__(self, vae_class, init_p=None, training=False, training_hp=None):
        self.vae_class = vae_class
        self.init_p = init_p
        self.training = training
        self.training_hp = training_hp


class ModelOptions():
    
    def __init__(self, model_class, init_p=None, training=0, training_p=None, training_hp=None, eval_p=None, requires_representation=False):
        self.model_class = model_class
        self.init_p = init_p
        self.training = training
        self.training_p = training_p
        self.training_hp = training_hp
        self.eval_p = eval_p
        self.requires_representation = requires_representation
        
        
class Options():
    
    def __init__(self, dataset, vae_options, model_options):
        self.dataset = dataset
        self.vae_options = vae_options
        self.model_options = model_options
        

dataset_options = {
    'CLEVR': CLEVRDataset,
    'CIFAR10': CIFARDataset
}

vae_options = {
    'CLEVR': VAEOptions(MONet, init_p={}, training_hp={}),
    'CIFAR10': VAEOptions(
        CIFARConvAE, 
        training=True,
        init_p={'feature_dim':32, 'code_dim':64}, 
        training_hp={'beta':0.1, 'iterations':20, 'batch_size':20, 'lr': 1e-4}
    )
}

model_options = {
    'MCHN': ModelOptions(MCHN, init_p={}, training_p={}, training_hp={}, eval_p={'beta':100}),
    'MemN2N': ModelOptions(
        MemN2N, training=1, 
        init_p={}, 
        training_p={}, 
        training_hp={'iterations':20, 'lr':0.0001, 'batch_size':100}, 
        eval_p={'alpha':100}
    ),
    'NTM': ModelOptions(
        NTM, 
        training=1, 
        init_p={}, 
        training_p={}, 
        training_hp={'iterations':20, 'lr':0.0001, 'batch_size':100}, 
        eval_p={}
    ),
    'GMM': ModelOptions(GMM, init_p={}, training_p={}, training_hp={}, eval_p={'sigma':0.1}),
    'BP-GMM': ModelOptions(
        BPGMM, 
        requires_representation=True, 
        init_p={}, 
        training_p={}, 
        training_hp={}, 
        eval_p={'sigma':0.1, 'lr':0.5, 'gamma':0.005}
    ),
    'MemN2N_star': ModelOptions(
        MemN2N, 
        training=2, 
        init_p={}, 
        training_p={}, 
        training_hp={'iterations':20, 'lr':0.0001, 'batch_size':100}, 
        eval_p={'alpha':100}
    ),
    'NTM_star': ModelOptions(
        NTM, 
        training=2, 
        init_p={}, 
        training_p={}, 
        training_hp={'iterations':20, 'lr':0.01, 'batch_size':100}, 
        eval_p={}
    ),
    'GMM_star': ModelOptions(
        GMMStar, 
        training=2, 
        init_p={}, 
        training_p={'iterations':1}, 
        training_hp={'iterations':20, 'lr':0.01, 'batch_size':100}, 
        eval_p={}
    )
}












