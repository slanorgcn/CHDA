
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn import GraphConv

class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        # 添加一个输出层，用于链接预测
        self.predict = nn.Linear(hidden_feats * 2, 1)
    
    def forward(self, g, inputs):

        # print(g)
        # print(inputs)
        
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    def predict_links(self, h, edges):
        # 用于链接预测的辅助函数，edges为节点对的索引
        edge_h = torch.cat((h[edges[0]], h[edges[1]]), dim=1)
        # print('edge_h@predict_links')
        # print(edge_h)
        return torch.sigmoid(self.predict(edge_h))
    