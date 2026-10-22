"""The three model families compared in this assignment."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Shallow multi-layer perceptron: 784 -> hidden -> 10 (one hidden layer)."""

    def __init__(self, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """VGG-style CNN: two conv blocks with max-pooling, then global average pooling."""

    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        c = width
        self.features = nn.Sequential(
            conv_bn(1, c), conv_bn(c, c), nn.MaxPool2d(2),               # 28 -> 14
            conv_bn(c, 2 * c), conv_bn(2 * c, 2 * c), nn.MaxPool2d(2),   # 14 -> 7
            conv_bn(2 * c, 4 * c),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * c, 10))
        self.to(memory_format=torch.channels_last)  # ~15% faster convolutions on CPU

    def forward(self, x):
        return self.head(self.features(x.contiguous(memory_format=torch.channels_last)))


class ViT(nn.Module):
    """Vision Transformer (encoder only).

    The 28x28 image is cut into non-overlapping patches (4x4 -> 7x7 = 49 tokens),
    each patch is linearly embedded, a learnable [CLS] token and learnable
    position embeddings are added, and a stack of pre-norm Transformer encoder
    layers is applied. The [CLS] output is used for classification.
    """

    def __init__(self, patch=4, dim=64, depth=4, heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        n_tokens = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)            # (B, 49, dim)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def build_model(name, **kwargs):
    return {"mlp": MLP, "cnn": CNN, "vit": ViT}[name](**kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
