import torch
import math
import numpy as np

def self_attention_pytorch(Q,K,V):
    
    m,n = Q.shape
    sm_scale = 1.0/ math.sqrt(n)
    attn = torch.mm(Q,K.transpose(-2,-1)) * sm_scale  # [m,m]
    attn = attn.softmax(dim=-1)                        # row-wise softmax
    O = torch.mm(attn,V)                              # [m,n]
    return O

