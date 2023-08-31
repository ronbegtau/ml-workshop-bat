import os
import time

# import matplotlib.pyplot as plt
import tqdm as tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, repeat  # , reduce
from einops.layers.torch import Rearrange, Reduce

# SET SEEDS
import random
import numpy as np
import sys

EMITTER_ONE_HOT_MAP = {
    '111': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '120': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '201': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '202': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '203': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '204': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '205': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '207': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '208': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '210': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '211': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '213': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '214': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '215': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '216': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '218': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '220': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '221': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    '222': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    '223': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    '225': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    '226': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '228': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    '230': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    '231': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    '233': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

if len(sys.argv) > 1:
    print("SET SEED")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


# SET SEEDS

def get_all_classes(root):
    all_files = os.listdir(root)
    classes = set()
    for fp in all_files:
        addr = fp.split("-")[-1][:-4]
        classes.add(addr)
    classes = sorted(classes)
    return classes


class AudioDataset(Dataset):
    def __init__(self, root, classes=None, transform=None):
        self.root = root
        self.transform = transform
        if classes is None:
            self.classes = get_all_classes(root)
        else:
            self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        self.emitter_map = dict(EMITTER_ONE_HOT_MAP)
        for fp in os.listdir(root):
            file_id, start_frame, end_frame, emitter, addressee = fp.split("-")
            addressee = addressee[:-4]
            self.samples.append(
                (os.path.join(root, fp), torch.tensor(self.emitter_map[emitter]), self.class_to_idx[addressee]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, emitter, addressee = self.samples[idx]
        img = Image.open(fp)
        if self.transform:
            img = self.transform(img)
        return img, emitter, addressee


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )  # this breaks down the image in s1xs2 patches, and then flat them

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # prepending the cls token
        x += self.positions
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)  # queries, keys and values matrix
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)

        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  # sum over the third axis
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, L=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ViT(torch.nn.Module):
    def __init__(self,
                 emitter_one_hot_size: int = len(EMITTER_ONE_HOT_MAP),
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 480,
                 depth: int = 1,
                 n_classes: int = 1000,
                 use_emitter: bool = False,
                 **kwargs):
        super(ViT, self).__init__()
        self.depth = depth
        self.use_emitter = use_emitter
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoder = TransformerEncoder(self.depth, emb_size=emb_size, **kwargs)
        self.classification_head = ClassificationHead(emb_size, emitter_one_hot_size, self.use_emitter, n_classes)

    def forward(self, x, emitter):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x, emitter)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, emitter_one_hot_size: int = len(EMITTER_ONE_HOT_MAP),
                 use_emitter: bool = False, n_classes: int = 1000):
        super(ClassificationHead, self).__init__()
        self.use_emitter = use_emitter
        linear_input_size = emb_size

        self.reduce = Reduce('b n e -> b e', reduction='mean')
        self.layer_norm = nn.LayerNorm(emb_size)
        self.hidden = nn.Linear(emb_size + emitter_one_hot_size, emb_size + emitter_one_hot_size)
        if use_emitter:
            linear_input_size += emitter_one_hot_size
        self.output = nn.Linear(linear_input_size, n_classes)

    def forward(self, x, emitter):
        x = self.reduce(x)
        x = self.layer_norm(x)
        if self.use_emitter:
            x = self.hidden(torch.cat((x, emitter), dim=1))
        x = self.output(x)
        return x


def save_model(path, tid, model, model_metadata: ViT, opt, epoch, loss, acc, classes):
    # Additional information
    torch.save({
        'tid': tid,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'classes': classes,
        'num_of_classes': len(classes),
        'depth': model_metadata.depth,
        'use_emitter': model_metadata.use_emitter
    }, path)


PATH = "./vit-ckpts/vit-model-{}-{}.pt"

if __name__ == "__main__":
    depth = 1
    use_emitter = False
    num_epochs = 30
    batch_size = 4
    root = "../data/spectograms-1/train"

    tid = int(time.time())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("root path to dataset:", root)
    print("using device:", device)
    # device = torch.device("cpu")

    train_dataset = AudioDataset(root, transform=transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    curr_epoch = -1

    if len(sys.argv) < 2:
        vit_metadata = ViT(depth=depth, use_emitter=use_emitter, n_classes=len(train_dataset.classes))
        vit = nn.DataParallel(vit_metadata)
        vit.to(device)
        optimizer = optim.Adam(vit.parameters(), lr=1e-3)

    else:
        # load checkpoint
        ckpt_path = sys.argv[1]
        print("loading checkpoint:", ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        tid = checkpoint['tid']

        vit = ViT(n_classes=checkpoint['num_of_classes'])
        vit = nn.DataParallel(vit)

        vit.load_state_dict(checkpoint['model_state_dict'])
        vit.to(device)

        optimizer = optim.Adam(vit.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        curr_epoch = checkpoint['epoch']

    # train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.3, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()

    print("start training, tid is", tid)
    for epoch in range(curr_epoch + 1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        vit.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, emitters, labels in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            emitter = emitters.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = vit(inputs, emitters)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        scheduler.step(epoch_acc)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        save_model(PATH.format(tid, epoch), tid, vit, vit_metadata, optimizer, epoch, epoch_loss, epoch_acc, train_dataset.classes)
