# 📚 Chinese Historical Documents Assistant (CHDA)

## 中国历史文献推荐小助手

本项目基于 `DGL`（Deep Graph Library） 框架开发，旨在通过深度学习模型推荐相关的中国历史文献（期刊）。项目包含自采数据集、训练过程、评测过程以及一个示例应用，支持传统低维图（如 `Neo4j`）与超图（如 `HyperGraphDB`）的导出内容作为数据集进行训练。

## 快速开始

安装所需依赖：

```sh
pip install dgl torch gensim fasttext
```

> 如果是 N 卡，可考虑安装 GPU 版本（未测试），具体请参考文档：[DGL 开始指南](https://www.dgl.ai/pages/start.html)

## 数据集

### 数据准备

为验证模型效果，我们提供了一个包含历史文献引用信息的简单数据集（包含了 2000-2024 年中，发表在`近代史研究`、`历史研究`、`民俗研究`、`中国边疆史地研究`以及`中国农史`的所有文章内容以及其引用关系），您可以打开 `./data/papers.json` 查看。

每篇文献都分配有一个唯一标识符（`UUID`），以便识别。为了适应链接预测的下游任务，我们在数据集中直接使用了所有可能的正样本，并计算出相同数量的不存在的链接作为负样本，供模型训练和验证使用。

> 如果您有自己的传统图数据库，可参考 `./data/papers.json` 结构导出一份自己的数据集。
>
> 如果您有自己的超图数据库，可参考[数据建模](#数据建模)中的内容，进行降维后使用，效果相同。（即借助传统图中的节点模拟超边，最后转为对应格式的 `./data/papers.json` 中，每条超边都用一个独立“超边节点”表示，之后分别在 `papers` 与 `edges` 中妥善表达即可。）

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

### 数据建模

> 本步骤为借助已经采集好的数据集（一对多的情况，即`论文`对`引用集`的超图场景）看做超图形式，并进行“超边降维”操作，仅供参考。在实际情况中，如果您有自己的图数据库或超图数据库源，可参考本步骤导出制作您自己的 `./data/papers.json`。

我们将超图转换为标准图进行模型训练，以此来模拟文献之间的引用关系。例如，一篇文章引用多篇文献时，会通过一个额外的节点（超边）连接这篇文章和其引用的文献节点，有效地表示引用关系。

以下是使用 `Neo4j` (`5.x` 版本)创建模型结构的示例，您可以直接将处置好的 `MySQL` 的表导出为两个 `CSV` 文件，然后直接进行导入后查看。

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

**图谱可视化（模拟超边）**

![图谱可视化1](https://oss.v-dk.com/img/202402211851917.jpg)

![图谱可视化2](https://oss.v-dk.com/img/202402211851640.jpg)

## 训练与验证

运行训练脚本（重命名`.env.sample`为`.env`并按需调整），数据集有更新，请及时删除旧的`graph_data.bin`、`your_features.pt`和`uuid_to_index.pt`文件：

```sh
python train_dgl.py
```

每轮训练后支持输入 `UUID` 进行手工验证，也可以单独运行小应用进行实际测试。

首次运行需要远程下载`fasttext`中的 `cc.zh.300.bin` 向量模型，请耐心等待。

## 测试

**模型测试**

如测试 `UUID` 为 `8d899b63-8d5a-4b92-92e9-6b8d90ff1ebd` 的论文名为：`日本全面侵华前夕对华态度新探`

键入 `UUID` 后推荐前 `20` 个相关内容，其中包含了引用的内容，也给推出了其他不在引用关系范围但特征相似的其他论文（如第二条的`西安事变与日本的对华政策`，已知西安事变是在日本侵华前夕的1936年12月发生的一起重大政治事件。即印证模型中给的推断结果与原论文正相关）。

![测试 UUID 用例](https://oss.v-dk.com/img/202402221516668.jpg)

![探索相关靠前结果](https://oss.v-dk.com/img/202402221517633.jpg)

![为论文自身引用关系](https://oss.v-dk.com/img/202402221517437.jpg)

**测试评分**

`10` 轮下的训练打分，超参如下：

```ini
BATCH_SIZE=4096
LR=0.001
HIDDEN_FEATS=512
EPOCH_COUNT=20
SAVE_PER_EPOCH=1
NUM_LAYERS=3
DROP_OUT=0.5
NUM_HEADS=4
TOP_K=20
```

![测试评分](https://oss.v-dk.com/img/202402221556319.png)

## 小应用

### CLI
```sh
python predict_dgl.py
```

训练好模型后，可通过本文件进行命令行下的预测。

## 发展路线图

- [x] 数据采集包：提供一个简单好用的知网期刊采集工具（基于公开数据，无任何敏感、侵权或隐私内容）。
- [x] 文本特征预处理：使用自然语言处理技术将文本特征（如标题、摘要）转换为数值型特征向量。
- [x] 特征工程：对类别型特征（如单个或多个的作者名）使用独热编码，并设计基于引用数量的特征等。
- [x] 模型设计：设计图神经网络模型，整合节点特征和图结构信息。
- [ ] 开发配套的示例应用。
