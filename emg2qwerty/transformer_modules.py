from collections.abc import Sequence

import torch
from torch import nn
import numpy as np

def scaled_dot_product_attention(q, k, v, mask=None):
    # Q: FloatTensor of shape (bsz, q_len, d)
    # K, V: FloatTensor of shape (bsz, kv_len, d)
    # mask: optional, BoolTensor of shape (bsz, q_len, kv_len)
    # return: outputs, attention_weights
    #   outputs: FloatTensor of shape (bsz, q_len, d), output of attention module
    #   attention_weights: FloatTensor of shape (bsz, q_len, kv_len), attention weights between 0-1
    _, _, d = q.shape

    k = k.permute(0, 2, 1)
    attention_weights = torch.bmm(q, k) / np.sqrt(d)
    if mask is not None:
        attention_weights = attention_weights.masked_fill(~mask, -np.inf)

    attention_weights = torch.softmax(attention_weights, -1)

    outputs = torch.bmm(attention_weights, v)
    return outputs, attention_weights


def multi_head_attention(q, k, v, num_heads: int, mask=None):
    # Q: FloatTensor of shape (bsz, q_len, d_model)
    # K, V: FloatTensor of shape (bsz, kv_len, d_model)
    # mask: optional, BoolTensor of shape (bsz, q_len, kv_len)
    # return: outputs, attention_weights
    #   outputs: FloatTensor of shape (bsz, q_len, d_model), output of attention module
    #   attention_weights: FloatTensor of shape (bsz, num_heads, q_len, kv_len), attention weights between 0-1
    
    assert q.shape[-1] % num_heads == 0
    d = q.shape[-1] // num_heads
    outputs = []
    attention_weights = []

    for head in range(num_heads):
        q_head = q[:, :, head*d:(head+1)*d]
        k_head = k[:, :, head*d:(head+1)*d]
        v_head = v[:, :, head*d:(head+1)*d]
        head_outputs, head_attention_weights = scaled_dot_product_attention(q_head, k_head, v_head, mask=mask)

        outputs.append(head_outputs)
        attention_weights.append(head_attention_weights)

    outputs = torch.cat(outputs, -1)
    attention_weights = torch.stack(attention_weights, dim=1)
    #print("attention_weights:", attention_weights.shape)
    return outputs, attention_weights

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None):
        q = self.q_proj(x)  # Note: for actual implementation, we need to do projection
        k = self.k_proj(x)
        v = self.v_proj(x)
        o, _ = multi_head_attention(q, k, v, self.num_heads, mask=mask)
        return self.o_proj(o)

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class PositionalEncodings(nn.Module):
    def __init__(self, d_model, base=10000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            base - Base for rotary positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, x):
        # x: FloatTensor of shape (bsz, seq_len)
        # return: pe, FloatTensor of shape (bsz, seq_len, d_model)
        #   pe[..., i] is the positional encoding for i-th position
        bsz, seq_len, channels = x.shape
        
        pe = []
        for i in range(self.d_model):
            pe_i = torch.arange(0, seq_len, dtype=torch.float)
            if i % 2 == 0:
                pe_i = torch.sin(pe_i / (self.base ** (2*i / self.d_model)))
            else:
                pe_i = torch.cos(pe_i / (self.base ** (2*i / self.d_model)))
            pe.append(pe_i)

        pe = torch.stack(pe, dim=-1)
        #print("position shape:", pe.shape)
        return pe.to(x.device)
    
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.0, charset_size=99, base=10000):
        super().__init__()
        self.pe = PositionalEncodings(d_model, base)  # positional embeddings
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dim_feedforward=2 * d_model, dropout=dropout) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, charset_size)#, bias=False)

    def forward(self, tokens, mask=None):
        x = tokens + self.pe(tokens)
        for l in self.layers:
            x = l(x, mask=mask)
        logits = self.lm_head(x)
        return logits