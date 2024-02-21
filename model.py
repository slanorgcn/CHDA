
import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import GraphConv

class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GNNModel, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = GraphConv(in_feats, hidden_feats).to(self.device)
        self.conv2 = GraphConv(hidden_feats, hidden_feats).to(self.device)
        # 添加一个输出层，用于链接预测
        self.predict = nn.Linear(hidden_feats * 2, 1).to(self.device)
    
    def forward(self, g, inputs):

        # print(g)
        # print(inputs)
        inputs = inputs.to(self.device)
        g = g.to(self.device)
        
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

    def predict_links(self, h, edges):
        # 用于链接预测的辅助函数，edges为节点对的索引
        edges = edges.to(self.device)
        edge_h = torch.cat((h[edges[0]], h[edges[1]]), dim=1)
        return torch.sigmoid(self.predict(edge_h))
    