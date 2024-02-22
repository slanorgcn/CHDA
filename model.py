import torch.nn as nn
import torch.nn.functional as F
import torch

# from dgl.nn import GraphConv
from dgl.nn import GATConv


class GNNModel(nn.Module):

    def __init__(self, in_feats, hidden_feats, num_layers, dropout_rate, num_heads):
        super(GNNModel, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  # 初始化dropout层

        # 第一个图卷积层
        self.layers.append(
            GATConv(in_feats, hidden_feats, num_heads=num_heads).to(self.device)
        )
        # 添加更多的图卷积层
        for i in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_feats * num_heads, hidden_feats, num_heads=num_heads).to(
                    self.device
                )
            )
        # 最后一个图卷积层，假设输出层的特征维度与隐藏层相同
        self.layers.append(
            GATConv(hidden_feats * num_heads, hidden_feats, num_heads=num_heads).to(
                self.device
            )
        )

        # 添加一个输出层，用于链接预测
        self.predict = nn.Linear(hidden_feats * num_heads * 2, 1).to(self.device)

    def forward(self, g, inputs):
        inputs = inputs.to(self.device)
        g = g.to(self.device)
        h = inputs
        for conv in self.layers:
            h = conv(g, h)
            h = F.relu(h)
            h = self.dropout(h)  # 应用dropout
        return h

    def predict_links(self, h, edges):
        # 用于链接预测的辅助函数，edges为节点对的索引
        edges = edges.to(self.device)
        edge_h = torch.cat((h[edges[:, 0]], h[edges[:, 1]]), dim=1)
        return torch.sigmoid(self.predict(edge_h))
