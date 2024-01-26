import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor, # (b, seq_len, dim)
    ) -> Tensor: # (b, seq_len, dim)
        residual = x
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # (b, seq_len, dim * 3)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v)
        )

        x = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(x, 'b h n d -> b n (h d)') # (b, seq_len, dim)
        out = self.to_out.forward(out)

        return out + residual
    
class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x: Tensor, # (seq_len, dim)
    ) -> Tensor: # (seq_len, dim)
        residual = x
        x = self.fc1.forward(x)
        x = F.gelu(x)
        return self.fc2.forward(x) + residual
    
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_hidden_dim: int,
    ):
        super().__init__()
        self.attn = Attention(dim, heads=heads)
        self.mlp = Mlp(dim, mlp_hidden_dim)
    
    def forward(
        self,
        x: Tensor, # (seq_len, dim)
    ) -> Tensor:
        x = self.attn.forward(x)
        x = self.mlp.forward(x)
        return x


class PPP(nn.Module):

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        heads: int,
        mlp_hidden_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks

        self.clap_proj = nn.Linear(512, dim)
        self.spec_proj = nn.Linear(768, dim)
        self.desc_proj = nn.Linear(768, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads, mlp_hidden_dim) for _ in range(num_blocks)
        ])

        self.out_proj = nn.Linear(dim, 1)

    def forward(
        self,
        image_embeddings: Tensor, # (b, n, 512)
        text_desc_embedding: Tensor, # (b, 1, 768)
        text_spec_embedding: Tensor, # (b, 1, 768)
    ) -> Tensor:
        
        image_embeddings = self.clap_proj.forward(image_embeddings) # (b, n, dim)
        text_desc_embedding = self.spec_proj.forward(text_desc_embedding) # (b, 1, dim)
        text_spec_embedding = self.desc_proj.forward(text_spec_embedding) # (b, 1, dim)

        x = torch.cat([
            image_embeddings,
            text_desc_embedding,
            text_spec_embedding,
        ], dim=1) # (b, n + 2, dim)

        for block in self.blocks:
            block: Block
            x = block.forward(x)
        
        # avg pool
        x = x.mean(dim=1) 

        # out proj
        x = self.out_proj.forward(x).squeeze(1)

        return x

    
    def loss(
        self,
        image_embeddings: Tensor, # (b, n, 512)
        text_desc_embedding: Tensor, # (b, 1, 768)
        text_spec_embedding: Tensor, # (b, 1, 768)
        target: Tensor, # (b, 1)
    ) -> Tensor:
        logits = self.forward(
            image_embeddings,
            text_desc_embedding,
            text_spec_embedding,
        )
        return F.mse_loss(logits, target)
