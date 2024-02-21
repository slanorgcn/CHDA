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


def load_model(model_path, in_feats, hidden_feats, num_layers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """加载训练好的模型"""
    model = GNNModel(in_feats=in_feats, hidden_feats=hidden_feats, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    return model