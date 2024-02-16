import json
import torch
import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


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
        return torch.sigmoid(self.predict(edge_h))
    
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    edges_src = [int(edge['source']) for edge in data['edges'] for target in edge['target']]
    edges_dst = [int(target) for edge in data['edges'] for target in edge['target']]
    
    num_papers = len(data['papers'])
    g = dgl.graph((edges_src, edges_dst))
    g.add_nodes(num_papers - g.number_of_nodes())  # 确保图中有正确数量的节点(补全无边节点)
    g = dgl.add_self_loop(g)
    
    years = np.array([[paper['year']] for paper in data['papers']])
    print('years')
    print(years)
    features = torch.FloatTensor(years)
    # todo 增加其他向量特征
    # features = torch.cat([torch.FloatTensor(years), title_embeddings, abstract_embeddings, author_features], dim=1)
    # print('features')
    # print(features)
    print('years.shape')
    print(years.shape)

    
    return g, features, data['papers'], data['edges']

def construct_negative_edges(g, num_neg_samples):
    import random
    
    # 获取所有可能的节点对作为候选
    all_nodes = list(range(g.number_of_nodes()))
    neg_edges = []
    tried_pairs = set()
    while len(neg_edges) < num_neg_samples:
        # 随机选择两个不同的节点
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u == v or (u, v) in tried_pairs or (v, u) in tried_pairs:
            continue  # 如果选择了相同的节点或已经尝试过这对节点，则跳过
        
        tried_pairs.add((u, v))
        tried_pairs.add((v, u))  # 注意添加两个方向，确保不重复选择
        
        # 检查这对节点是否已经连接
        if not g.has_edges_between(u, v):
            neg_edges.append((u, v))
            if len(neg_edges) % 100 == 0:
                print(f"Generated {len(neg_edges)} negative edges so far...")
    
    # 转换为Tensor
    neg_edges = torch.tensor(neg_edges).t()  # 转置以匹配DGL的边索引格式
    return neg_edges


def recommend_papers(model, g, features, paper_id, top_k=10):
    model.eval()
    with torch.no_grad():
        paper_embeddings = model(g, features)
        query_embedding = paper_embeddings[paper_id]
        scores = torch.matmul(paper_embeddings, query_embedding)
        top_k_adjusted = min(top_k, g.number_of_nodes() - 2)  # 确保不超出范围
        _, indices = torch.topk(scores, k=top_k_adjusted+1)  # +1可能包括论文自身
    recommended_ids = [idx.item() for idx in indices if idx.item() != paper_id][:top_k_adjusted]
    return recommended_ids

def main():
    g, features, papers, edges = load_data('paper.json')
    
    print('g.number_of_nodes()')  # 图中节点数量
    print(g.number_of_nodes())  # 图中节点数量
    print('features.shape[0]')  # 特征张量中的样本数（节点数）
    print(features.shape[0])  # 特征张量中的样本数（节点数）

    model = GNNModel(in_feats=features.shape[1], hidden_feats=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # print(papers)
    # print(edges)
    
    # 构造正样本边
    pos_edges = []
    for edge in edges:
        src = int(edge['source'])
        targets = edge['target']
        for dst in targets:
            pos_edges.append((src, int(dst)))

    pos_edges = torch.tensor(pos_edges).t()
    
    pos_edges = torch.tensor(pos_edges).t()
    # print(pos_edges.size())  # 检查尺寸是否正确
    neg_edges = construct_negative_edges(g, len(pos_edges))
    
    # print(pos_edges)
    # print(neg_edges)
    
    for epoch in range(1000):
        model.train()
        h = model(g, features)
        pos_score = model.predict_links(h, pos_edges)
        neg_score = model.predict_links(h, neg_edges)
        
        # 使用二进制交叉熵损失
        labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])
        predictions = torch.cat([pos_score, neg_score])
        
        # 亦可↓ 
        # Ensure the labels and predictions have the same shape
        # labels = labels.view(predictions.shape)
        # 在计算损失之前，确保labels的尺寸与predictions匹配
        labels = labels.unsqueeze(1)
        
        loss = F.binary_cross_entropy(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 从外部获取paper_id，这里仅作演示，实际使用时应由用户输入或其他方式获取
    input_paper_id = int(input("请输入论文ID："))

    recommended_ids = recommend_papers(model, g, features, paper_id=input_paper_id, top_k=10)
    print("为论文ID {} 推荐的相关论文ID列表:".format(input_paper_id), recommended_ids)

if __name__ == '__main__':
    main()
