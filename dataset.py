import torch
from torch.utils.data import Dataset


class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, dictionary, glove, args):
        # Unknown Token is index 1 (<UNK>)
        self.x = [[dictionary.get(token,1) for token in token_list] for token_list in texts]
        self.embeds = [[glove[token] for token in token_list] for token_list in texts]
        self.y = labels

    def __len__(self):
        """Return the data length"""
        return len(self.x)

    def __getitem__(self, idx):
        """Return one item on the index"""
        return self.x[idx], self.embeds[idx], self.y[idx]


def collate_fn(data, args, pad_idx=0):
    """Padding"""
    texts, embeds, labels = zip(*data)
    texts = [s + [pad_idx] * (args.max_len - len(s)) if len(s) < args.max_len else s[:args.max_len] for s in texts]
    
    embed = None
    if args.lexical:
        for text_embed in embeds:
            for token_embed in text_embed:
                if embed != None:
                    embed = torch.cat(embed,token_embed.unsqueeze(0)))
                else:
                    embed = token_embed.unsqueeze(0)
    return torch.LongTensor(texts), embed, torch.LongTensor(labels)