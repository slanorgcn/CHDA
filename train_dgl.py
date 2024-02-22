import os
import json
import torch
import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import fasttext.util

from dgl.nn import GraphConv
from dgl.data.utils import save_graphs, load_graphs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from model import GNNModel
import config
import utils


def load_data(filename):

    # 如果特征与图数据有更新，请删除以下文件，确保删除旧的graph_data.bin、your_features.pt和uuid_to_index.pt文件。
    # 这样，下次运行load_data函数时，会根据更新后的数据重新生成和保存这些文件。
    graph_data_file = "graph_data.bin"
    features_file = "features_file.pt"
    uuid_to_index_file = "uuid_to_index.pt"

    # 每次都载入内存
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 检查文件是否存在，如果存在则直接加载
    if (
        os.path.exists(graph_data_file)
        and os.path.exists(features_file)
        and os.path.exists(uuid_to_index_file)
    ):
        graphs, _ = load_graphs(graph_data_file)
        g = graphs[0]  # 加载图
        features_scaled = torch.load(features_file)  # 加载特征
        uuid_to_index = torch.load(uuid_to_index_file)  # 加载uuid到索引的映射
        return g, features_scaled, data["papers"], data["edges"], uuid_to_index

    # 如果文件不存在，执行原有的数据加载和处理流程

    # 加载预训练的fastText模型
    fasttext.util.download_model("zh", if_exists="ignore")  # 'zh'代表中文
    ft = fasttext.load_model("cc.zh.300.bin")

    # 创建从UUID到整数索引的映射
    uuid_to_index = {paper["id"]: idx for idx, paper in enumerate(data["papers"])}

    # 使用映射转换edges中的UUID到整数索引
    edges_src = [
        uuid_to_index[edge["source"]]
        for edge in data["edges"]
        for target in edge["target"]
    ]
    edges_dst = [
        uuid_to_index[target] for edge in data["edges"] for target in edge["target"]
    ]

    # print('uuid_to_index')
    # print(uuid_to_index)
    # print('edges_src')
    # print(edges_src)
    # print('edges_dst')
    # print(edges_dst)

    num_papers = len(data["papers"])
    g = dgl.graph((edges_src, edges_dst))
    g.add_nodes(
        num_papers - g.number_of_nodes()
    )  # 确保图中有正确数量的节点(补全无边节点)
    g = dgl.add_self_loop(g)

    # 保存一份DGL的二进制格式用于预测脚本
    save_graphs(graph_data_file, [g])

    # 处理年份特征
    years = np.array([[paper["year"]] for paper in data["papers"]])

    # 将字符串类型转换为浮点数
    years = years.astype(np.float32)

    # 然后转换为torch.FloatTensor
    years = torch.FloatTensor(years)

    # 生成标题和摘要和期刊的词向量
    title_embeddings = np.array(
        [ft.get_sentence_vector(paper["title"]) for paper in data["papers"]]
    )
    abstract_embeddings = np.array(
        [
            ft.get_sentence_vector(paper["abstract"].replace("\n", ""))
            for paper in data["papers"]
        ]
    )
    journal_embeddings = np.array(
        [ft.get_sentence_vector(paper["journal"]) for paper in data["papers"]]
    )

    # 处理作者信息（简单示例：使用OneHotEncoder）
    authors_list = [
        ",".join(paper["authors"]) for paper in data["papers"]
    ]  # 将作者列表转换为字符串
    encoder = OneHotEncoder(sparse=False)
    author_features = encoder.fit_transform(np.array(authors_list).reshape(-1, 1))

    # 拼接所有特征
    features = np.concatenate(
        [
            years,
            title_embeddings,
            abstract_embeddings,
            journal_embeddings,
            author_features,
        ],
        axis=1,
    )
    # features = np.concatenate([years], axis=1)
    features = torch.FloatTensor(features)

    # 实例化归一化工具
    scaler = StandardScaler()

    # 对年份特征进行归一化
    years_scaled = scaler.fit_transform(years)

    # 对嵌入向量进行归一化
    title_embeddings_scaled = scaler.fit_transform(title_embeddings)
    abstract_embeddings_scaled = scaler.fit_transform(abstract_embeddings)
    journal_embeddings_scaled = scaler.fit_transform(journal_embeddings)

    # 拼接所有归一化后的特征
    features_scaled = np.concatenate(
        [
            years_scaled,
            title_embeddings_scaled,
            abstract_embeddings_scaled,
            journal_embeddings_scaled,
            author_features,
        ],
        axis=1,
    )
    features_scaled = torch.FloatTensor(features_scaled)

    # 注意：作者特征(author_features)通常不需要归一化，因为它是独热编码的

    # 保存特征数据为 .pt 文件用于预测脚本
    # torch.save(features, features_file)
    torch.save(features_scaled, features_file)

    # 保存从UUID到图节点索引的映射数据为 .pt 文件用于预测脚本
    uuid_to_index = {paper["id"]: idx for idx, paper in enumerate(data["papers"])}
    torch.save(uuid_to_index, uuid_to_index_file)

    # 将映射转换为JSON格式的字符串
    uuid_to_index_json = json.dumps(uuid_to_index, indent=4)

    # 将JSON字符串保存到文件，以备参考对照
    with open("uuid_to_index.json", "w") as json_file:
        json_file.write(uuid_to_index_json)

    # print('title_embeddings')
    # print(title_embeddings)

    # print('author_features')
    # print(author_features)

    # return g, features, data['papers'], data['edges'], uuid_to_index
    return g, features_scaled, data["papers"], data["edges"], uuid_to_index


def construct_positive_negative_edges(
    g, edges, uuid_to_index, test_size=0.2, val_size=0.1
):
    # 构造正样本边，使用uuid_to_index映射
    pos_edges = []
    for edge in edges:
        src = uuid_to_index[edge["source"]]
        targets = [uuid_to_index[t] for t in edge["target"]]
        for dst in targets:
            pos_edges.append((src, dst))

    # 构造负样本
    all_nodes = list(range(g.number_of_nodes()))
    neg_edges = []
    tried_pairs = set()

    while len(neg_edges) < len(pos_edges):
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u == v or (u, v) in tried_pairs:
            continue

        tried_pairs.add((u, v))
        tried_pairs.add((v, u))

        if not g.has_edges_between([u], [v]).bool().item():
            neg_edges.append((u, v))

    # print('pos_edges@construct_positive_negative_edges')
    # print(pos_edges[:10])
    # print('neg_edges@construct_positive_negative_edges')
    # print(neg_edges[:10])

    # 分割数据集
    pos_train, pos_temp = train_test_split(
        pos_edges, test_size=test_size + val_size, random_state=42
    )
    pos_val, pos_test = train_test_split(
        pos_temp, test_size=test_size / (test_size + val_size), random_state=42
    )

    neg_train, neg_temp = train_test_split(
        neg_edges, test_size=test_size + val_size, random_state=42
    )
    neg_val, neg_test = train_test_split(
        neg_temp, test_size=test_size / (test_size + val_size), random_state=42
    )

    # 转换为Tensor
    pos_train_tensor = torch.tensor(pos_train, dtype=torch.long)
    pos_val_tensor = torch.tensor(pos_val, dtype=torch.long)
    pos_test_tensor = torch.tensor(pos_test, dtype=torch.long)
    neg_train_tensor = torch.tensor(neg_train, dtype=torch.long)
    neg_val_tensor = torch.tensor(neg_val, dtype=torch.long)
    neg_test_tensor = torch.tensor(neg_test, dtype=torch.long)

    return (
        pos_train_tensor,
        pos_val_tensor,
        pos_test_tensor,
        neg_train_tensor,
        neg_val_tensor,
        neg_test_tensor,
    )


def prepare_dataloader(pos_edges, neg_edges, batch_size):

    # print('pos_edges@prepare_dataloader')
    # print(pos_edges[:10])
    # print('neg_edges@prepare_dataloader()')
    # print(neg_edges[:10])

    # 创建DataLoader
    dataset = TensorDataset(pos_edges, neg_edges)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def train(model, train_loader, optimizer, g, features):
    device = model.device  # 从模型中获取当前设备
    model.train()
    total_loss = 0
    for batch in train_loader:

        # print('batch')
        # print(batch)

        pos_edges, neg_edges = batch

        pos_edges = pos_edges.to(device)  # 移动到正确的设备
        neg_edges = neg_edges.to(device)  # 移动到正确的设备

        optimizer.zero_grad()

        # 获取图的节点表示
        h = model(g, features)

        # print('pos_edges@train')
        # print(pos_edges)

        # print('neg_edges@train')
        # print(neg_edges)

        # 使用predict_links方法计算正负样本的预测得分
        pos_score = model.predict_links(h, pos_edges)
        neg_score = model.predict_links(h, neg_edges)

        # 创建标签并计算损失
        labels = torch.cat(
            [
                torch.ones(pos_score.size(0), device=device),
                torch.zeros(neg_score.size(0), device=device),
            ]
        )
        predictions = torch.cat([pos_score, neg_score])

        # 在计算损失之前，确保labels的尺寸与predictions匹配
        labels = labels.unsqueeze(1)

        loss = F.binary_cross_entropy(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Average Loss: {avg_loss}")
    return avg_loss


def evaluate(model, loader, g, features):
    device = next(model.parameters()).device  # 更健壮的获取模型所在设备的方式

    model.eval()
    y_true = []
    y_pred_probs = []  # 存储预测概率
    with torch.no_grad():
        for batch in loader:
            pos_edges, neg_edges = batch
            pos_edges, neg_edges = pos_edges.to(device), neg_edges.to(device)

            # 获取图的节点表示
            h = model(g.to(device), features.to(device))

            # 使用predict_links方法计算正负样本的预测得分
            pos_score = model.predict_links(h, pos_edges)
            neg_score = model.predict_links(h, neg_edges)

            # 更新真实标签和预测得分列表
            y_true.extend([1] * pos_score.size(0))
            y_true.extend([0] * neg_score.size(0))
            y_pred_probs.extend(pos_score.squeeze().tolist())
            y_pred_probs.extend(neg_score.squeeze().tolist())

    # 假设y_pred_probs已经是概率，无需再次应用sigmoid
    y_pred = [
        1 if prob > 0.5 else 0 for prob in y_pred_probs
    ]  # 根据阈值将概率转换为二值预测结果

    # 计算性能指标
    auc = roc_auc_score(y_true, y_pred_probs)  # AUC计算应使用预测概率
    accuracy = accuracy_score(y_true, y_pred)  # 准确率使用二值化后的预测

    print(f"Evaluation - AUC: {auc}, Accuracy: {accuracy}")
    return auc, accuracy


def main():

    # 确定执行的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    g, features, papers, edges, uuid_to_index = load_data("./data/paper.json")

    # 移动图和特征到指定设备
    g = g.to(device)
    features = features.to(device)

    pos_train, pos_val, pos_test, neg_train, neg_val, neg_test = (
        construct_positive_negative_edges(g, edges, uuid_to_index)
    )

    train_loader = prepare_dataloader(pos_train, neg_train, config.batch_size)
    val_loader = prepare_dataloader(pos_val, neg_val, config.batch_size)
    test_loader = prepare_dataloader(pos_test, neg_test, config.batch_size)

    # print('g.number_of_nodes()')  # 图中节点数量
    # print(g.number_of_nodes())  # 图中节点数量
    # print('features.shape[0]')  # 特征张量中的样本数（节点数）
    # print(features.shape[0])  # 特征张量中的样本数（节点数）

    model = GNNModel(
        in_feats=features.shape[1],
        hidden_feats=config.hidden_feats,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 尝试加载已有模型
    model_checkpoint_path = "model_checkpoint.pth"
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        print("Model loaded and will continue training.")
    else:
        print("No existing model found. Starting training from scratch.")

    for epoch in range(config.epoch_count):

        train_loss = train(model, train_loader, optimizer, g, features)
        val_auc, val_accuracy = evaluate(model, val_loader, g, features)
        print(
            f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        # 训练结束后保存模型:% N
        # 每N轮保存一次模型
        if (epoch + 1) % config.save_per_epoch == 0:
            torch.save(model.state_dict(), model_checkpoint_path)
            print(f"Model saved at epoch {epoch+1}")

    test_auc, test_accuracy = evaluate(model, test_loader, g, features)
    print(f"Test AUC: {test_auc:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # # 快捷推理：从外部获取UUID形式的paper_id
    # input_uuid = input("请输入论文UUID：")
    # if input_uuid in uuid_to_index:
    #     paper_index = uuid_to_index[input_uuid]
    #     recommended_ids = utils.recommend_papers(model, g, features, paper_index, top_k=10)

    #     # 将推荐的索引转换回UUID
    #     index_to_uuid = {idx: paper['id'] for idx, paper in enumerate(papers)}
    #     recommended_uuids = [index_to_uuid[idx] for idx in recommended_ids]

    #     print("为论文UUID {} 推荐的相关论文UUID列表:".format(input_uuid), recommended_uuids)
    # else:
    #     print("输入的UUID未找到对应的论文。")


if __name__ == "__main__":
    main()
