"""分割网络的另一种结构：DGCNN 局部特征 + 全局自注意力（arch = "dgcnn_attn"）。

DGCNN 只有一个最大池化的全局向量，每个点不知道“牙弓上别处有几颗牙”，而牙位编号、缺牙恰恰要靠这个。
这里从采样点里取一部分当锚点，锚点之间做几层自注意力（看到整个牙弓），再让每个点对锚点做一次交叉注意力取回上下文。
输入输出与 deep.build_model 相同：(B, C, N) → (B, n_classes, N)。训练和推理时点的顺序是随机采样的，所以前 A 个点就是随机锚点。
"""

from __future__ import annotations


def build(in_dim: int, n_classes: int, k: int = 20, dim: int = 256, anchors: int = 512, layers: int = 3, heads: int = 8):
    import torch

    nn = torch.nn

    def knn(x, k):  # (B, C, N) → (B, N, k)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]

    def edge_features(x, idx):
        b, c, n = x.shape
        flat = (idx + torch.arange(b, device=x.device).view(-1, 1, 1) * n).view(-1)
        xt = x.transpose(2, 1).contiguous()
        neigh = xt.view(b * n, -1)[flat].view(b, n, -1, c)
        center = xt.view(b, n, 1, c).expand(-1, -1, neigh.shape[2], -1)
        return torch.cat([neigh - center, center], dim=3).permute(0, 3, 1, 2).contiguous()

    class EdgeConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.mlp = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

        def forward(self, x, idx):
            return self.mlp(edge_features(x, idx)).max(dim=-1)[0]

    class DGCNNAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.anchors = k, anchors
            self.e1, self.e2, self.e3 = EdgeConv(in_dim, 96), EdgeConv(96, 128), EdgeConv(128, 256)
            self.local = nn.Sequential(nn.Conv1d(480, dim, 1, bias=False), nn.BatchNorm1d(dim), nn.LeakyReLU(0.2))
            self.pos = nn.Sequential(nn.Linear(in_dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim))
            layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.cross = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Sequential(
                nn.Conv1d(2 * dim, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Conv1d(128, n_classes, 1),
            )

        def forward(self, x):  # (B, C, N)
            idx = knn(x[:, :3], self.k)
            x1 = self.e1(x, idx)
            x2 = self.e2(x1, knn(x1, self.k))
            x3 = self.e3(x2, knn(x2, self.k))
            local = self.local(torch.cat([x1, x2, x3], dim=1))  # (B, dim, N)
            tok = local.transpose(1, 2) + self.pos(x.transpose(1, 2))  # (B, N, dim)
            ctx = self.encoder(tok[:, : self.anchors])  # 锚点之间：整个牙弓的上下文
            got = self.cross(self.norm(tok), ctx, ctx, need_weights=False)[0]  # 每个点取回上下文
            return self.head(torch.cat([local, got.transpose(1, 2)], dim=1))

    return DGCNNAttn()
