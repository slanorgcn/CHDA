import torch
import json

from model import GNNModel
import torch.nn.functional as F

import config


# 点积计算容易推荐为固定内容，不推荐
def recommend_papers(model, g, features, paper_id, top_k):
    model.eval()
    with torch.no_grad():
        # print('paper_id@recommend_papers:', paper_id)
        paper_embeddings = model(g, features)
        query_embedding = paper_embeddings[paper_id]
        # print('query_embedding@recommend_papers:', query_embedding)
        scores = torch.matmul(paper_embeddings, query_embedding)
        # print('scores@recommend_papers:', scores)
        # print(scores.size())

        # log
        # 将映射转换为JSON格式的字符串
        # 转换为Python列表
        _scores, _indices = torch.topk(scores, top_k)
        scores_list = _scores.tolist()
        indices_list = _indices.tolist()
        # 创建一个列表，每个元素是(score, index)对
        recommendations = list(zip(scores_list, indices_list))
        # 将JSON字符串保存到文件，以备参考对照
        with open("./data/topk.json", "w") as json_file:
            json.dump(recommendations, json_file, indent=4)

        top_k_adjusted = min(top_k, g.number_of_nodes() - 1)  # 确保不超出范围
        _, indices = torch.topk(scores, k=top_k_adjusted + 1)  # +1可能包括论文自身

    recommended_ids = [idx.item() for idx in indices if idx.item() != paper_id][
        :top_k_adjusted
    ]
    return recommended_ids


# 推荐余弦相似度
def recommend_papers_cosine_similarity(model, g, features, paper_id, top_k):
    model.eval()
    with torch.no_grad():
        paper_embeddings = model(g, features)
        query_embedding = paper_embeddings[paper_id].unsqueeze(
            0
        )  # 增加一个维度以便广播
        scores = F.cosine_similarity(paper_embeddings, query_embedding, dim=1)
        top_k_adjusted = min(top_k, g.number_of_nodes() - 1)  # 确保不超出范围
        _, indices = torch.topk(scores, k=top_k_adjusted + 1)  # +1可能包括论文自身
    recommended_ids = [idx.item() for idx in indices if idx.item() != paper_id][
        :top_k_adjusted
    ]
    return recommended_ids


def load_model(model_path, in_feats, hidden_feats, num_layers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """加载训练好的模型"""
    model = GNNModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        num_layers=num_layers,
        dropout_rate=config.dropout_rate,
        num_heads=config.num_heads,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    return model
