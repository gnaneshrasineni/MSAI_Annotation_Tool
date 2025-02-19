import torch
import torch.nn as nn

from segment_anything import sam_model_registry, SamPredictor

class MedSAM(nn.Module):
    def __init__(self, 
                 checkpoint: str = './MedSAM/medsam_vit_b.pth',
                 model_type: str = 'vit_b',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        super(MedSAM, self).__init__()
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)