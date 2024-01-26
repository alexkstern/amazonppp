import torch
from model import PPP

ckpt_path = './checkpoints/ckpt_1000.pth'
# load state dict
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

model = PPP(
    dim=512,
    num_blocks=6,
    heads=8,
    mlp_hidden_dim=512,
)

model.load_state_dict(ckpt)

image_embeddings = torch.randn(1, 10, 512)
text_desc_embedding = torch.randn(1, 1, 768)
text_spec_embedding = torch.randn(1, 1, 768)

price = model.forward(image_embeddings, text_desc_embedding, text_spec_embedding)
print(price)