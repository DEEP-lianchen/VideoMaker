import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union
from einops import rearrange, repeat
try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e: 
    xformers_available = False
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention import FeedForward
xformers_available = False
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    
def process_latent_tensor(tensor): 
    # concat self attention with last frame
    # [b, f, t, c]
    _, f, t, _ = tensor.shape
    n_visual_with_condition = f*t
    last_frame = tensor[:, -1:, :, :]  # [b, 1, t, c]
    repeated_last_frame = last_frame.repeat(1, f-1, 1, 1)  # [b, f-1, t, c]
    remaining_frames = tensor[:, :-1, :, :]  #[b, f-1, t, c]
    n_visual = remaining_frames.shape[2]
    result = torch.cat((remaining_frames, repeated_last_frame), dim=2) #[b, f-1, 2*t, c]
    result = rearrange(result, 'b f t c -> (b f) t c').contiguous() 
    return result, n_visual


class PersonalizedInjectionAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_frames=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        num_frames=self.num_frames
        residual = hidden_states # [b*f,h*w,c]
        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c',f=num_frames+1)

        hidden_states, n_visual = process_latent_tensor(hidden_states) # [b*(f-1),h*w*2,c]

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, scale=attn.scale
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        

        hidden_states = rearrange(hidden_states, '(b f) t c -> b f t c', f=num_frames)
        hidden_states, ref_frame_result = hidden_states[:,:,:n_visual,:], hidden_states[:,:,n_visual:,:]
        ref_frame_result = torch.mean(ref_frame_result, dim=1, keepdim=True)
        hidden_states = torch.cat((hidden_states, ref_frame_result), dim=1)
        hidden_states = rearrange(hidden_states, 'b f t c -> (b f) t c')
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states