import torch.nn as nn
import torch.nn.functional as F
import torch

# from dgl.nn import GraphConv
from dgl.nn import GATConv


class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, dropout_rate, num_heads):
        super(GNNModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化ModuleList来存储GAT层
        self.layers = nn.ModuleList()

        # 初始化dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 添加第一个GAT层
        self.layers.append(
            GATConv(
                in_feats=in_feats,
                out_feats=hidden_feats,
                num_heads=num_heads,
                feat_drop=dropout_rate,
                attn_drop=dropout_rate,
            ).to(self.device)
        )

        # 添加中间层
        for _ in range(1, num_layers - 1):
            self.layers.append(
                GATConv(
                    in_feats=hidden_feats * num_heads,
                    out_feats=hidden_feats,
                    num_heads=num_heads,
                    feat_drop=dropout_rate,
                    attn_drop=dropout_rate,
                ).to(self.device)
            )

        # 添加最后一个GAT层，注意这里没有使用多头机制，以减少输出维度
        self.layers.append(
            GATConv(
                in_feats=hidden_feats * num_heads,
                out_feats=hidden_feats,
                num_heads=1,
                feat_drop=dropout_rate,
                attn_drop=dropout_rate,
            ).to(self.device)
        )

        # 链接预测的线性层，考虑到最后一个GAT层的输出维度
        self.predict = nn.Linear(2 * hidden_feats, 1).to(self.device)

    def forward(self, g, inputs):
        h = inputs.to(self.device)
        g = g.to(self.device)
        for conv in self.layers:
            # 注意：对于GATConv，输出是(num_heads, N, D)形状的张量，需要调整形状
            h = conv(g, h).flatten(1)  # 将多头输出扁平化处理
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def predict_links(self, h, edges):
        edges = edges.to(self.device)
        # 由于最后一个GAT层输出的是(N, D)，所以这里不需要修改维度处理
        edge_h = torch.cat((h[edges[:, 0]], h[edges[:, 1]]), dim=1)
        return torch.sigmoid(self.predict(edge_h))
