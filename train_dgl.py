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
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    ndcg_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate
from scipy import stats

from model import GNNModel
import config
import utils
import shutil
import time


def load_data(filename):

    # 如果特征与图数据有更新，请删除以下文件，确保删除旧的graph_data.bin、features_file.pt和uuid_to_index.pt文件。
    # 这样，下次运行load_data函数时，会根据更新后的数据重新生成和保存这些文件。
    graph_data_file = "./data/graph_data.bin"
    features_file = "./data/features_file.pt"
    uuid_to_index_file = "./data/uuid_to_index.pt"

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
    with open("./data/uuid_to_index.json", "w") as json_file:
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


def ndcg_score(y_true, y_scores, k=20):
    actual = np.take(y_true, np.argsort(y_scores)[::-1])[:k]
    ideal = np.sort(y_true)[::-1][:k]
    dcg = np.sum(actual / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0


def evaluate(model, loader, g, features, k):
    device = next(model.parameters()).device

    model.eval()
    y_true = []
    y_scores = []  # 用于NDCG计算的分数

    with torch.no_grad():
        for batch in loader:
            pos_edges, neg_edges = batch
            pos_edges, neg_edges = pos_edges.to(device), neg_edges.to(device)

            h = model(g.to(device), features.to(device))
            pos_score = model.predict_links(h, pos_edges)
            neg_score = model.predict_links(h, neg_edges)

            y_true.extend([1] * len(pos_score) + [0] * len(neg_score))
            y_scores.extend(
                pos_score.squeeze().cpu().numpy().tolist()
                + neg_score.squeeze().cpu().numpy().tolist()
            )

    auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, [1 if score > 0.5 else 0 for score in y_scores])
    recall = recall_score(y_true, [1 if score > 0.5 else 0 for score in y_scores])
    f1 = f1_score(y_true, [1 if score > 0.5 else 0 for score in y_scores])
    ndcg = ndcg_score(np.array(y_true), np.array(y_scores), k=k)

    print(
        f"Evaluation - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, NDCG@{k}: {ndcg:.4f}"
    )
    return auc, accuracy, recall, f1, ndcg


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

    model = GNNModel(
        in_feats=features.shape[1],
        hidden_feats=config.hidden_feats,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        num_heads=config.num_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 尝试加载已有模型
    model_checkpoint_path = "./model/model_checkpoint.pth"
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        print("Model loaded and will continue training.")

        # Create a backup of the existing model checkpoint
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        backup_path = f"./model/backup/model_checkpoint-{timestamp}.pth"
        shutil.copyfile(model_checkpoint_path, backup_path)
        print(f"Model checkpoint backed up as {backup_path}")
    else:
        print("No existing model found. Starting training from scratch.")

    best_val_auc = 0.0  # 假设以AUC为基准进行早停
    patience_counter = 0
    patience = config.patience  # 设置耐心值，例如10个epoch

    training_logs = [
        [
            "Epoch",
            "Loss",
            "Val AUC",
            "Val Accuracy",
            "Val Recall",
            "Val F1",
            "Val NDCG@" + str(config.top_k),
            "备注（Comments）",
        ]
    ]

    for epoch in range(config.epoch_count):
        train_loss = train(model, train_loader, optimizer, g, features)
        auc, accuracy, recall, f1, ndcg = evaluate(
            model, val_loader, g, features, config.top_k
        )

        print(
            f"🤖 Epoch {epoch+1}, Current AUC: {auc:.4f}, Best AUC: {best_val_auc:.4f}, Patience Counter: {patience_counter}"
        )

        if auc > best_val_auc:
            best_val_auc = auc
            patience_counter = 0  # 重置耐心计数器
            torch.save(model.state_dict(), model_checkpoint_path)  # 保存最好的模型
            print(
                f"🎉 Model saved at epoch {epoch+1} with improvement in AUC: {auc:.4f}"
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"❌ Early stopping triggered at epoch {epoch+1}")
            break  # 提前终止训练

        # 动态添加备注
        if epoch < config.epoch_count * 0.25:
            comment = "初次训练（Initial Training）"
        elif epoch < config.epoch_count * 0.5:
            comment = "性能提升（Performance Improvement）"
        elif epoch < config.epoch_count * 0.75:
            comment = "接近收敛（Approaching Convergence）"
        else:
            comment = "训练结束（End of Training）"

        training_logs.append(
            [
                epoch + 1,
                f"{train_loss:.4f}",
                f"{auc:.4f}",
                f"{accuracy:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{ndcg:.4f}",
                comment,
            ]
        )

    # 训练结束后，输出所有训练日志
    print(tabulate(training_logs, headers="firstrow", tablefmt="grid"))

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(model_checkpoint_path))
    print("Loaded best model for testing.")

    auc, accuracy, recall, f1, ndcg = evaluate(
        model, test_loader, g, features, config.top_k
    )
    print(
        f"💡 Test AUC: {auc:.4f}, Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}, Test NDCG@{str(config.top_k)}: {ndcg:.4f}"
    )


if __name__ == "__main__":
    main()
