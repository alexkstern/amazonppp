import torch
from torch.utils.data import Dataset
import numpy as np

class PPPDataset(Dataset):
    def __init__(
        self,
        pad_length: int = 16,
    ):
        self.image_embeddings = np.load('./data/image_embeddings.npy') # (32180, 512)
        self.text_embeddings = np.load('./data/text_embeddings_sentence_transformers.npy')
        self.metadata = np.load('./data/metadata.npy', allow_pickle=True)

        self.pad_length = pad_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata_row = self.metadata[idx]
        image_embeddings = self.image_embeddings[metadata_row['image_indices']]
        text_desc_embedding = self.text_embeddings[metadata_row['text_desc_index']]
        text_spec_embedding = self.text_embeddings[metadata_row['text_spec_index']]
        price = float(metadata_row['price'])

        # make them tensors
        image_embeddings = torch.from_numpy(image_embeddings).float()
        text_desc_embedding = torch.from_numpy(text_desc_embedding).unsqueeze(0).float()
        text_spec_embedding = torch.from_numpy(text_spec_embedding).unsqueeze(0).float()
        price = torch.tensor(price).float()

        # pad image_embeddings to pad_length - 2
        image_embeddings = torch.nn.functional.pad(image_embeddings, (0, 0, 0, self.pad_length - 2 - image_embeddings.shape[0]))

        # Return the preprocessed sample
        return dict(
            image_embeddings=image_embeddings,
            text_desc_embedding=text_desc_embedding,
            text_spec_embedding=text_spec_embedding,
            target=price,
        )

if __name__ == '__main__':
    dataset = PPPDataset()
    print(dataset[0])