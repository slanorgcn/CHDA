import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN, HGNNP
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

# 定义训练函数
def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()  # 将模型设置为训练模式

    st = time.time()  # 开始计时
    optimizer.zero_grad()  # 清除之前的梯度
    outs = net(X, A)  # 前向传播，X是特征矩阵，A是超图结构
    outs, lbls = outs[train_idx], lbls[train_idx]  # 选取训练集的输出和标签
    loss = F.cross_entropy(outs, lbls)  # 计算交叉熵损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()

# 定义推断函数
@torch.no_grad()  # 不计算梯度，节省计算资源
def infer(net, X, A, lbls, idx, test=False):
    net.eval()  # 将模型设置为评估模式
    outs = net(X, A)  # 前向传播
    outs, lbls = outs[idx], lbls[idx]  # 选取相应数据集的输出和标签
    if not test:
        res = evaluator.validate(lbls, outs)  # 验证
    else:
        res = evaluator.test(lbls, outs)  # 测试
    return res

if __name__ == "__main__":
    set_seed(2021)  # 设置随机种子
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 检查是否可以使用GPU
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])  # 初始化评估器
    data = Cooking200()  # 加载Cooking200数据集

    # 初始化特征矩阵为单位矩阵，标签，以及超图结构
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    G = Hypergraph(data["num_vertices"], data["edge_list"])
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # 初始化模型和优化器
    net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=True)  # 使用批归一化
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    # 将数据和模型转移到GPU（如果可用）
    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None  # 用于保存最佳模型状态
    best_epoch, best_val = 0, 0
    for epoch in range(200):  # 训练200个epoch
        # 训练
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        # 验证
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)
            if val_res > best_val:  # 保存最佳模型
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())

    # 在这里保存最佳模型状态
    torch.save(best_state, "best_model_state.pth")
                
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # 测试
    print("test...")
    net.load_state_dict(best_state)  # 加载最佳模型状态
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
