import torch
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UpDecoderBlock2D
from typing import Optional

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

# WA: added graph break between self.resnets and self.upsamplers fixed blank output
UpDecoderBlock2D_original_forward = UpDecoderBlock2D.forward
def gaudi_UpDecoderBlock2D_forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
    for resnet in self.resnets:
        hidden_states = resnet(hidden_states, temb=temb)

    # add graph break
    if torch.compiler.is_dynamo_compiling():
        torch._dynamo.graph_break()

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)

    return hidden_states
UpDecoderBlock2D.forward = gaudi_UpDecoderBlock2D_forward
