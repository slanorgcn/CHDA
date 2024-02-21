# 📚 Chinese Historical Documents Assistant (CHDA)

## 中国历史文献推荐小助手

本项目基于 `DGL`（Deep Graph Library） 框架开发，旨在通过深度学习模型推荐相关的中国历史文献（期刊）。项目包含自采数据集、训练过程、评测过程以及一个示例应用，支持传统低维图（如 `Neo4j`）与超图（如 `HyperGraphDB`）的导出作为数据集进行训练。

## 快速开始

安装所需依赖：

```sh
pip install dgl torch gensim fasttext
```

如果是 N 卡，可考虑安装 GPU 版本，具体请参考文档：[DGL 开始指南](https://www.dgl.ai/pages/start.html)

## 数据集

### 数据准备

为了准备数据集，我们需要收集文章及其引用信息，并确保每篇文章都有一个唯一的标识符，如 `UUID` 或 `DOI` 号。我们提供了一个包含历史文献引用信息的简单数据集（包含了 2000-2024 年中，发表在`近代史研究`、`历史研究`、`民俗研究`、`中国边疆史地研究`以及`中国农史`的所有文章内容以及其引用关系），您可以打开 `./data/papers.json` 查看。

每篇文献都分配有一个唯一标识符（`UUID`），以便识别。为了适应链接预测的下游任务，我们在数据集中直接使用了所有可能的正样本，并计算出相同数量的不存在的链接作为负样本，供模型训练和验证使用。

> 本项目支持超图数据训练。当前本训练集为低维测试数据进行直接训练，并在传统图数据库中借助虚拟节点表示为超边进行示例。
>
> 如果您有自己的超图数据库，也可参考**二部图建模**中的内容，进行降维后使用，效果相同。
>
> 即借助节点表达超边，最后转为对应格式的 `./data/papers.json` 中，每条超边都用一个独立超边节点表示，在 `papers` 与 `edges` 中妥善表达即可。

### 数据采集

数据集的采集小工具附在 `tools/cnkiCrawler.py` 中，可通过更改环境变量自行采集（信息均来自知网公开内容，不涉及特殊内容，采用非登录状态的合法方式采集，并降低了采集频率，以达到模拟自然浏览的使用状态），导入 `cnkiCrawler.sql` 并配置好后相关参数（重命名`.env.sample`为`.env`）后，可运行下方命令进行采集。

```sh
python ./tools/run_chda.py
```

### 数据导出

```sh
python ./tools/export_json.py
```

## 主要功能

### 特征处理

我们目前提供了基于 `fasttext` 的文本特征处理方法和基本的独热编码（`one-hot`）技术。在实际应用中，您可能需要采用更先进的文本表示技术，如 `TF-IDF`、`Word2Vec` 或 `BERT`。

### 二部图建模

我们将超图转换为标准图进行模型训练，以此来模拟文献之间的引用关系。例如，一篇文章引用多篇文献时，会通过一个额外的节点（超边）连接这篇文章和其引用的文献节点，有效地表示引用关系。

以下是使用 `Neo4j` (`5.x` 版本)创建模型结构的示例，您可以直接将处置好的 `MySQL` 的表导出为两个 `CSV` 文件，然后直接进行导入后查看：

**导入节点（论文）**

```cypher
LOAD CSV WITH HEADERS FROM 'file:///papercollection.csv' AS row
CREATE (:Paper {
  uuid: row.uuid,
  title: row.paper_title,
  year: row.publication_year,
  journal: row.journal_name,
  authors: row.authors,
  abstract: row.abstract
})
```

**导入关系（引用关系）**

对于 `paperreferences.csv` 中的每一行，我们创建一个中间节点（模拟超边）来表示一篇论文引用的所有其他论文，并创建相应的关系。由于 `paperreferences.csv` 可能包含一对多的引用关系，我们将按照源论文（`paper_uuid`）进行分组，并为每个源论文创建一个单独的引用集合节点。

```cypher
LOAD CSV WITH HEADERS FROM 'file:///paperreferences.csv' AS row
MATCH (source:Paper {uuid: row.paper_uuid})
WITH source, COLLECT(row.referenced_paper_uuid) AS refs
CREATE (refSet:ReferenceSet {id: randomUUID()})
MERGE (source)-[:HAS_REF_SET]->(refSet)
WITH refSet, refs
UNWIND refs AS refUuid
MATCH (target:Paper {uuid: refUuid})
MERGE (refSet)-[:REFERENCES]->(target);
```

**二部图截图（模拟超边）**

![alt text](https://oss.v-dk.com/img/202402211851917.jpg)

![alt text](https://oss.v-dk.com/img/202402211851640.jpg)

## 训练与验证

运行训练脚本（重命名`.env.sample`为`.env`并按需调整），数据集有更新，请及时删除旧的`graph_data.bin`、`your_features.pt`和`uuid_to_index.pt`文件：

```sh
python train_dgl.py
```

每轮训练后支持输入 `UUID` 进行手工验证，也可以单独运行小应用进行实际测试。

首次运行需要远程下载`fasttext`中的 `cc.zh.300.bin` 向量模型，请耐心等待。

## 小应用

```sh
python predict_dgl.py
```

训练好模型后，可通过本文件进行命令行下的预测。

## 发展路线图

- [x] 数据采集包：提供一个简单好用的知网期刊采集工具（基于公开数据，无任何敏感、侵权或隐私内容）。
- [x] 文本特征预处理：使用自然语言处理技术将文本特征（如标题、摘要）转换为数值型特征向量。
- [x] 特征工程：对类别型特征（如单个或多个的作者名）使用独热编码，并设计基于引用数量的特征等。
- [x] 模型设计：设计图神经网络模型，整合节点特征和图结构信息。
- [x] 开发配套的示例应用。
