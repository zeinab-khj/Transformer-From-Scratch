import torch
from torch.utils.data import Dataset

# Special Tokens
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

class ToyTranslationDataset(Dataset):

    def __init__(self, num_samples=100):

        self.data = []


        for i in range(num_samples):

            src = torch.tensor(
                [1, 2, 3, 4],
                dtype=torch.long
            )


            tgt = torch.tensor(
                [START_TOKEN, 5, 6, 7, 8و END_TOKEN],
                dtype=torch.long
            )


            self.data.append(
                (src, tgt)
            )


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        src, tgt = self.data[index]

        return {
            "src": src,
            "tgt": tgt
        }
