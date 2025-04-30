import torch
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UpDecoderBlock2D

CrossAttnDownBlock2D_original_forward = CrossAttnDownBlock2D.forward
@torch.compiler.disable
def gaudi_CrossAttnDownBlock2D_forward(self, *args, **kwargs):
    return CrossAttnDownBlock2D_original_forward(self, *args, **kwargs)
CrossAttnDownBlock2D.forward = gaudi_CrossAttnDownBlock2D_forward

CrossAttnUpBlock2D_original_forward = CrossAttnUpBlock2D.forward
@torch.compiler.disable
def gaudi_CrossAttnUpBlock2D_forward(self, *args, **kwargs):
    return CrossAttnUpBlock2D_original_forward(self, *args, **kwargs)
CrossAttnUpBlock2D.forward = gaudi_CrossAttnUpBlock2D_forward

UpDecoderBlock2D_original_forward = UpDecoderBlock2D.forward
@torch.compiler.disable
def gaudi_UpDecoderBlock2D_forward(self, *args, **kwargs):
    return UpDecoderBlock2D_original_forward(self, *args, **kwargs)
UpDecoderBlock2D.forward = gaudi_UpDecoderBlock2D_forward
