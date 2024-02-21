import torch
import dgl
import json
import os

from tabulate import tabulate

from model import GNNModel
import config

def load_model(model_path, in_feats, hidden_feats, num_layers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """加载训练好的模型"""
    model = GNNModel(in_feats=in_feats, hidden_feats=hidden_feats, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    return model

def predict_link(model, g, features, src_node, dst_node):
    """预测给定一对节点之间链接的存在概率"""
    with torch.no_grad():
        h = model(g, features)
        src_index = g.nodes().tolist().index(src_node)
        dst_index = g.nodes().tolist().index(dst_node)
        score = model.predict_links(h, torch.tensor([[src_index], [dst_index]]))
        return torch.sigmoid(score).item()

def recommend_papers(model, g, features, paper_id, top_k=10):
    model.eval()
    with torch.no_grad():
        paper_embeddings = model(g, features)
        query_embedding = paper_embeddings[paper_id]
        scores = torch.matmul(paper_embeddings, query_embedding)
        top_k_adjusted = min(top_k, g.number_of_nodes() - 1)  # 确保不超出范围
        _, indices = torch.topk(scores, k=top_k_adjusted+1)  # +1可能包括论文自身
    recommended_ids = [idx.item() for idx in indices if idx.item() != paper_id][:top_k_adjusted]
    return recommended_ids

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    # print(torch.cuda.get_device_name(0))
    # print(torch.__version__)
    print(f"Using device: {device}")
    
    # 加载图数据和特征（这里需要你根据实际情况填充或修改）
    g = dgl.load_graphs('graph_data.bin')[0][0]  # 假设图数据已保存为DGL的二进制格式
    features = torch.load('features_file.pt')  # 假设节点特征已保存为PyTorch张量
    
    g = g.to(device)
    features = features.to(device)

    # 加载模型（调整路径、输入特征维度和隐藏层特征维度）
    model_path = 'model_checkpoint.pth'
    in_feats = features.shape[1]
    model = load_model(model_path, in_feats, config.hidden_feats, config.num_layers)

    # 定义UUID到索引的映射
    with open('./data/paper.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        uuid_to_index = {paper['id']: idx for idx, paper in enumerate(data['papers'])}

    while True:
        # 从用户输入获取UUID
        input_uuid = input("请输入论文UUID（输入'exit'退出）：")
        if input_uuid.lower() == 'exit':
            break
        else:
            # 查找输入UUID对应的论文索引
            if input_uuid in uuid_to_index:
                paper_index = uuid_to_index[input_uuid]
                recommended_ids = recommend_papers(model, g, features, paper_index, top_k=10)

                # 获取推荐论文的详细信息
                recommended_papers_info = []
                for idx in recommended_ids:
                    paper_info = data['papers'][idx]
                    recommended_papers_info.append([paper_info['id'], paper_info['title'], ", ".join(paper_info['authors']), paper_info['year'], paper_info['journal']])

                # 打印推荐列表
                headers = ["UUID", "标题", "作者", "出版社", "年份"]
                print(f"\n为论文UUID {input_uuid} 推荐的相关论文列表:")
                print(tabulate(recommended_papers_info, headers=headers, tablefmt="grid"))
            else:
                print("输入的UUID未找到对应的论文。")

if __name__ == '__main__':
    main()
