import json
import torch
import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.model_selection import train_test_split
import fasttext.util
from sklearn.preprocessing import OneHotEncoder


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
    # 加载预训练的fastText模型
    fasttext.util.download_model('zh', if_exists='ignore')  # 'zh'代表中文
    ft = fasttext.load_model('cc.zh.300.bin')
    
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # 创建从UUID到整数索引的映射
    uuid_to_index = {paper['id']: idx for idx, paper in enumerate(data['papers'])}
    
    # 使用映射转换edges中的UUID到整数索引
    edges_src = [uuid_to_index[edge['source']] for edge in data['edges'] for target in edge['target']]
    edges_dst = [uuid_to_index[target] for edge in data['edges'] for target in edge['target']]
    
    # print('uuid_to_index')
    # print(uuid_to_index)
    # print('edges_src')
    # print(edges_src)
    # print('edges_dst')
    # print(edges_dst)
    
    num_papers = len(data['papers'])
    g = dgl.graph((edges_src, edges_dst))
    g.add_nodes(num_papers - g.number_of_nodes())  # 确保图中有正确数量的节点(补全无边节点)
    g = dgl.add_self_loop(g)
    
    # 处理年份特征
    years = np.array([[paper['year']] for paper in data['papers']])
    years = torch.FloatTensor(years)
    
    # 生成标题和摘要的词向量
    title_embeddings = np.array([ft.get_sentence_vector(paper['title']) for paper in data['papers']])
    abstract_embeddings = np.array([ft.get_sentence_vector(paper['abstract']) for paper in data['papers']])
    
    # 处理作者信息（简单示例：使用OneHotEncoder）
    authors_list = [",".join(paper['authors']) for paper in data['papers']]  # 将作者列表转换为字符串
    encoder = OneHotEncoder(sparse=False)
    author_features = encoder.fit_transform(np.array(authors_list).reshape(-1, 1))
    
    # 拼接所有特征
    features = np.concatenate([years, title_embeddings, abstract_embeddings, author_features], axis=1)
    features = torch.FloatTensor(features)
    
    # print('title_embeddings')
    # print(title_embeddings)
    
    # print('author_features')
    # print(author_features)
    
    return g, features, data['papers'], data['edges'], uuid_to_index

def construct_negative_edges(g, num_neg_samples):
    import random
    
    all_nodes = list(range(g.number_of_nodes()))
    neg_edges = []
    tried_pairs = set()
    
    while len(neg_edges) < num_neg_samples:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u == v or (u, v) in tried_pairs:
            continue
        
        tried_pairs.add((u, v))
        tried_pairs.add((v, u))
        
        if not g.has_edges_between(u, v):
            neg_edges.append([u, v])
    
    neg_edges = torch.tensor(neg_edges)
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
    g, features, papers, edges, uuid_to_index = load_data('paper.json')
    
    # print('g.number_of_nodes()')  # 图中节点数量
    # print(g.number_of_nodes())  # 图中节点数量
    # print('features.shape[0]')  # 特征张量中的样本数（节点数）
    # print(features.shape[0])  # 特征张量中的样本数（节点数）

    model = GNNModel(in_feats=features.shape[1], hidden_feats=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # print(papers)
    # print(edges)
    
    # 构造正样本边，使用uuid_to_index映射
    pos_edges = []
    for edge in edges:
        src = uuid_to_index[edge['source']]
        targets = [uuid_to_index[t] for t in edge['target']]
        for dst in targets:
            pos_edges.append((src, dst))
    pos_edges = torch.tensor(pos_edges).t()

    # print(pos_edges.size())  # 检查尺寸是否正确
    neg_edges = construct_negative_edges(g, len(pos_edges))
    
    # print('pos_edges')
    # print(pos_edges)
    # print('neg_edges')
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

    # 从外部获取UUID形式的paper_id
    input_uuid = input("请输入论文UUID：")
    if input_uuid in uuid_to_index:
        paper_index = uuid_to_index[input_uuid]
        recommended_ids = recommend_papers(model, g, features, paper_index, top_k=10)
        
        # 将推荐的索引转换回UUID
        index_to_uuid = {idx: paper['id'] for idx, paper in enumerate(papers)}
        recommended_uuids = [index_to_uuid[idx] for idx in recommended_ids]
        
        print("为论文UUID {} 推荐的相关论文UUID列表:".format(input_uuid), recommended_uuids)
    else:
        print("输入的UUID未找到对应的论文。")


if __name__ == '__main__':
    main()
