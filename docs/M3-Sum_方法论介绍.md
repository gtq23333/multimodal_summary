# M3-Sum 方法论介绍

> 本文档为论文「方法」章节初稿，基于 `trial_31` 项目实现与可复现实验结果撰写。  
> Stage-2 数字来源：`outputs_copy/trial_31/eval/stage2_reranking_summary.csv`。

---

## 3 方法

### 3.1 任务定义

本文研究**数学建模竞赛论文的多模态摘要生成**（mathematical modeling competition paper multimodal summarization）。与传统科学文献摘要仅压缩文本不同，该任务要求系统从一篇包含正文、公式、图表及图注的长文档中，生成由文本片段与关键图片共同构成的图文融合摘要。其目标不是复现全文图片，而是在有限阅读时间内呈现论文的关键问题、建模思路、求解流程与主要结果，使评阅者能够快速建立对论文方法贡献与结果可信度的整体认识。

该任务与**科学多模态摘要**（scientific multimodal summarization）相近：后者通常同时包含文本摘要生成、代表性图片选择与图文语义匹配等子任务 [1]。但数学建模竞赛论文具有额外特点：其方法结构通常围绕赛题的多个子问题展开，图片在论文中的作用也呈现明显的功能差异，例如问题分析图、模型框架图、算法流程图、实验结果图与局部敏感性分析图。因而，单纯依赖图像—查询的语义相似度，难以稳定判断一张图片是否适合作为「摘要图片」。

形式化地，给定赛题文本 P 与一篇论文


\mathcal{D}=(\mathcal{B},\mathcal{F}),


其中 \mathcal{B}=b_1,\ldots,b_M 为按文档顺序排列的文本块，\mathcal{F}=f_1,\ldots,f_N 为论文中的候选图片集合；每张图片 f_i 具有图像内容、图注 c_i、正文位置 p_i 与可能的图号引用信息。系统输出图文摘要


\mathcal{S}=\big[(t_1), (v_1), (t_2),\ldots,(v_L),(t_{L+1})\big],


其中 t_l 是文本片段，v_l\in \mathcal{F} 是插入在对应位置的关键图片。本文将其分解为三个阶段：


P \xrightarrow{\text{query planning}} \mathcal{Q}
\xrightarrow{\text{text evidence retrieval + figure reranking}} \mathcal{F}_K
\xrightarrow{\text{multimodal generation}} \mathcal{S}.


其中，\mathcal{Q} 为由赛题驱动的子查询集合，\mathcal{F}_K 为重排后保留的 Top-K 图片候选池。

需要强调的是，本文的「RAG」并非面向开放网络或超大外部知识库的检索；它更准确地属于**文档内证据检索增强生成**（intra-document evidence-grounded generation）：系统在单篇长论文内部定位与赛题信息需求相关的正文证据，再利用这些证据约束文本摘要与图片选择。该设计继承了 RAG 将参数化生成与显式非参数记忆结合的思想 [2]，但将知识源限定为待摘要论文自身，从而避免引入论文未声明的外部事实。

---



### 3.2 数据集构建与多模态摘要标注

我们构建了面向数学建模竞赛论文的图文摘要数据集。当前主实验集 `trial_31` 包含 **31 篇**优秀竞赛论文，每篇论文均以结构化 Markdown、正文图片和图注的形式保存。为保证候选图片具有可解释的论文语境，候选池默认限定为**正文中带有图注的图片**，即采用 `body_with_caption` 过滤策略。

对于每篇论文，标注者首先保留原始摘要文本，并在人工编辑界面中从候选图片中选择适合进入摘要的图片；随后，标注者确定图片插入位置，并构造交替的文本—图片序列。标注结果表示为


\mathcal{Y} = \left(A,\mathcal{I},\mathcal{M}\right),


其中 A 为参考摘要文本，\mathcal{I}=(h_j,\pi_j)_{j=1}^{R} 为图片插入集合，h_j 为图片哈希标识，\pi_j 为插入位置，\mathcal{M} 为包含文本段与图片段的多模态参考序列。由此可得到两种监督信号：


G_{\text{img}}=h_1,\ldots,h_R,


用于评估关键图片检索与重排；以及 \mathcal{M}，用于评估生成摘要的图片集合、顺序和插入位置。

该标注范式与 SMSMO 等科学多模态摘要工作中将文本生成、图片选择和图文匹配联合建模的设定一致 [1]，但本文的标注进一步保留了图片在摘要中的插入位置，以支持图文组织层面的评估。需要指出，当前 G_{\text{img}} 由摘要插图标注导出，因此应将其定义为 *summary-oriented figure relevance*，即「适合进入摘要的图片相关性」，而非论文中所有语义相关图片的穷尽标注。

**实现路径**：`data_annotation/` 标注工具 → `data/trial_31/ground_truth/`；配置见 `src/configs/trial_31.yaml`。

---



### 3.3 赛题驱动的查询规划与正文证据检索

数学建模论文的摘要信息需求不应仅由论文标题或原摘要决定，还应与其需要解决的赛题相联系。为此，本文首先使用大语言模型将赛题文本 P 分解为三个互补的信息需求维度：


\mathcal{Q}=q^{(a)},q^{(m)},q^{(s)},


分别对应**问题分析**（analysis）、**建模**（modeling）和**求解/验证**（solution）。每个子查询包含自然语言描述与关键词，用于寻找论文中回答相应赛题需求的文本证据。

对于任一子查询 q，我们在文本块集合 \mathcal{B} 上执行混合检索。具体地，将稀疏检索分数和稠密语义相似度归一化后线性融合：

# 
s_{\text{hyb}}(q,b)

\lambda s_{\text{BM25}}(q,b)
+
(1-\lambda)\cos(\mathbf e_q,\mathbf e_b),


其中 \mathbf e_q 与 \mathbf e_b 分别为查询和文本块的向量表示，\lambda=0.4。BM25 提供词项匹配与可解释的词法证据 [3]，稠密向量表示补充语义层面的匹配；最终保留每个子查询的 Top-P 文本块，当前实现中 P=20。

这一步的作用不是直接生成摘要，而是将「赛题要求」转化为可检索的正文证据，为后续图片重排提供问题条件化的语义锚点。

**实现路径**：`src/m3sum/stage1_query/query_builder.py`；混合检索见 `src/m3sum/stage2_rerank/hybrid_retriever.py`。

---



### 3.4 LG-JSSF：链接引导的结构化图片重排

#### 3.4.1 设计动机

通用视觉语言重排器通常以「文本查询—图片/图注」的相关性作为主要信号。然而，数学建模论文中的摘要选图不仅取决于语义相关性，还取决于图片在文档结构中的功能：早期出现的总体框架图可能比后部局部曲线图更适合摘要；被正文显式讨论的图片比孤立图片更具解释价值；方法图、流程图与系统结构图通常比重复性实验图更能降低读者理解成本。

因此，本文不将图片选择视为纯粹的跨模态相似度排序，而将其建模为融合直接语义、图文链接、文档结构与视觉表征先验的结构化相关性估计：

$$
\mathrm{Rel}(f \mid \mathcal{D})
=
\left[
\alpha_f S_{\mathrm{D}}(f)
+
(1-\alpha_f)S_{\mathrm{L}}(f)
\right]
\cdot P_{\mathrm{L}}(f)
\cdot P_{\mathrm{T}}(f)
+
\beta P_{\mathrm{C}}(f).
$$

其中，$f$ 表示候选图片，$\mathcal{D}$ 表示由正文块与图片集合构成的当前论文文档。$S_{\mathrm{D}}(f)$ 和 $S_{\mathrm{L}}(f)$ 分别表示直接语义分数与图文链接分数；$P_{\mathrm{L}}(f)$、$P_{\mathrm{T}}(f)$ 和 $P_{\mathrm{C}}(f)$ 分别表示布局、图片类型和视觉聚类先验；$\alpha_f$ 用于根据链接证据的可靠性调节直接语义与图文链接的相对贡献，$\beta$ 控制视觉聚类先验的加性贡献。

这种设计与信息检索中将非文本特征、位置和结构信息纳入相关性建模的思想相一致 [3]。同时，图文表示由预训练视觉—语言模型提供；例如，CLIP 类模型表明图文对比学习能够获得可迁移的跨模态表征 [4]。

#### 3.4.2 语义相关性与图文链接

对于候选图片，系统以图注作为其文本侧表示，并计算其与赛题驱动子查询的语义相关性。设 $\mathcal{Q}$ 是赛题分解得到的子查询集合，直接语义分数定义为：

$$
S_{\mathrm{D}}(f)
=
\max_{q\in\mathcal{Q}}
\cos(\mathbf{e}_q,\mathbf{e}_f).
$$

其中，$\mathbf{e}_q$ 表示子查询 $q$ 的文本向量，$\mathbf{e}_f$ 表示候选图片图注的文本向量。最大池化保留与图片最匹配的赛题维度，避免不同子问题之间的无关匹配相互稀释。

仅比较子查询与图注容易忽略图片所依赖的正文语境。因此，系统进一步计算图文链接分数 $S_{\mathrm{L}}(f)$：首先检索与赛题子查询相关的正文块，再将其与图片对应的证据块匹配。图片证据优先使用正文中的显式图号引用和图注关联块；仅当正文存在“如图所示”等指代线索时，才补充图片前后的局部上下文。匹配时，正文距离较远的块会被衰减。

显式图号引用被视为高可信证据，局部邻近上下文仅被视为弱证据。因此，$\alpha_f$ 采用预设的证据类型门控：存在可靠显式链接时，直接语义和链接分数等权融合；仅存在局部证据时，链接项降低权重；没有可靠链接时，系统退化为仅使用直接语义分数。该机制避免局部窗口中的偶然共现被误认为图片的重要性。具体阈值、衰减与截断参数作为实现细节在实验设置中统一固定。

#### 3.4.3 文档与视觉先验

布局先验 $P_{\mathrm{L}}(f)$ 反映图片在论文中的出现位置。令 $r_f$ 为图片图号；当图号缺失时，以其在正文中的出现顺序代替，则：

$$
P_{\mathrm{L}}(f)
=
\frac{1}{\log_2(1+r_f)}.
$$

该先验刻画数模论文中总体方法图、框架图和流程图通常较早出现的文档结构规律。类型先验 $P_{\mathrm{T}}(f)$ 由图注中的功能性关键词构建：对流程图、框架图和方法图给予适度提升，对局部重复性结果图施加轻微抑制。二者均作为软偏置参与排序，而不替代语义和链接证据。

为补充文本外的视觉信息，本文使用 Chinese-CLIP 对图片进行编码，并以图片与领域簇中心的匹配程度构造视觉聚类弱先验 $P_{\mathrm{C}}(f)$。该先验仅在图片具有较明确的簇归属时生效，并以较小的加性权重 $\beta$ 参与最终得分，以避免视觉类别先验覆盖直接语义证据。

综上，LG-JSSF 并非以单一余弦相似度决定摘要图片，而是将赛题相关性、正文图文关联、论文结构和视觉表征作为互补证据进行可解释融合。

**实现路径**：直接语义、图文链接与布局/类型先验见 `src/m3sum/stage2_rerank/reranker.py` 和 `co_occurrence.py`；视觉聚类先验见 `src/m3sum/stage2_rerank/cluster_prior.py`；融合逻辑见 `src/m3sum/stage2_rerank/fusion.py`。

---



### 3.5 图文摘要生成

重排后，系统选择 Top-K 图片作为生成候选池，并提供赛题问题、原摘要、混合检索证据、正文内容、候选图像及图注。本文实现两种生成策略：

1. **Text-RAG-then-Rewrite**：先基于正文证据生成文本摘要，再结合候选图片重写为图文摘要；
2. **End-to-End VLM**：将文本上下文与候选图片共同输入视觉语言模型，直接生成带图片占位符的图文摘要。

生成结果被规范化为 JSON，包含 `generated_summary`、`selected_image_hashes`、`placeholders` 和选图理由。图片在摘要中的插入形式为：


[\texttt{Insert Figure C}i],


其中 C_i 对应候选池中的第 i 张图片。当前 Stage-3 默认候选池大小 K=6，生成模型为 `qwen3.6-27b`。

**实现路径**：`src/m3sum/stage3_generation/generators.py`；配置见 `src/configs/trial_31.yaml` 中 `stage3_generation` 段。

---



### 3.6 实验观察与方法合理性

在 `trial_31` 的 31 篇人工标注论文上，启发式重排并非依赖单一手工规则，而是通过「问题驱动语义—图文链接—文档结构—视觉表征」四类互补信号估计摘要选图价值。实验显示，**布局先验是最关键的模块**：移除 P_{\text{layout}} 后，R-Precision 从约 **0.507** 降至约 **0.282**（约 −44%）。这说明数模论文的摘要选图确实具有不能由图像语义单独解释的结构规律。

在与通用基线的比较中（Stage-2 宏平均，`outputs_copy/trial_31/eval`）：


| 方法                          | R-Prec    | IP@3      | IR@3      | MAP       | MRR       |
| --------------------------- | --------- | --------- | --------- | --------- | --------- |
| **Proposed**                | **0.507** | **0.452** | **0.504** | **0.619** | **0.819** |
| Qwen3-VL-Rerank-ImgCap+Link | 0.507     | 0.441     | 0.494     | 0.586     | 0.715     |
| Layout-Order                | 0.465     | 0.398     | 0.432     | 0.582     | 0.879     |
| Qwen3-VL-Rerank-ImgCap      | 0.374     | 0.366     | 0.401     | 0.500     | 0.641     |


**可写结论**：

1. Proposed 与 Qwen3-VL-Rerank-ImgCap+Link 在 **R-Precision 上基本持平**，但在 **MAP、MRR、IP@3、IR@3** 等排序敏感或 Top-K 选择指标上表现更优。
2. 相对弱 neural 基线 Qwen3-VL-Rerank-ImgCap，Proposed 的 R-Precision 提升约 **+35%**（0.507 vs 0.374）。
3. Layout-Order 的 MRR 最高（0.879），进一步佐证文档顺序是数模论文选图的强先验，但单独依赖顺序不足以覆盖语义匹配需求。

该结果支持本文的核心判断：通用 VLM 重排可提供强跨模态语义匹配能力，但面向数学建模竞赛论文的摘要选图还需要显式利用文档位置、图文引用关系与图片功能属性。

**表述边界**（论文写作时须遵守）：

- 不宜宣称 Proposed 在所有 Stage-2 指标上全面超过 Qwen3-VL-Rerank-ImgCap+Link。
- 不宜宣称两阶段候选压缩一定优于强端到端 VLM 的全图输入生成；Stage-3 实验表明 All-Figures E2E 在 ref-based 综合分上可能更优。
- 宜将 LG-JSSF 表述为**可解释的、面向预算约束的候选压缩与排序机制**，而非强 VLM 的替代品。

---



## 参考文献

[1] Zhong, X., Tan, Z., Gao, S., Li, J., Shen, J., Ji, J.-Y., et al. *SMSMO: Learning to Generate Multimodal Summary for Scientific Papers*. Knowledge-Based Systems, 310, 112908, 2025. [https://doi.org/10.1016/j.knosys.2024.112908](https://doi.org/10.1016/j.knosys.2024.112908)

[2] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33, 9459–9474, 2020.

[3] Robertson, S., & Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389, 2009. [https://doi.org/10.1561/1500000019](https://doi.org/10.1561/1500000019)

[4] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. *Learning Transferable Visual Models from Natural Language Supervision*. Proceedings of the 38th International Conference on Machine Learning, 8748–8763, 2021.

[5] Khattab, O., & Zaharia, M. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 39–48, 2020. [https://doi.org/10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)

[6] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. *BERTScore: Evaluating Text Generation with BERT*. International Conference on Learning Representations, 2020.

[7] Bai, S., et al. *Qwen3-VL Technical Report*. arXiv:2511.21631, 2025. [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)

---



## 附录：与代码模块对应关系


| 论文章节             | 代码路径                                                      |
| ---------------- | --------------------------------------------------------- |
| 3.2 标注           | `data_annotation/` → `data/trial_31/ground_truth/`        |
| 3.3 Query 构建     | `src/m3sum/stage1_query/query_builder.py`                 |
| 3.3 混合检索         | `src/m3sum/stage2_rerank/hybrid_retriever.py`             |
| 3.4 LG-JSSF      | `src/m3sum/stage2_rerank/reranker.py`, `co_occurrence.py` |
| 3.4 ClusterPrior | `src/m3sum/stage2_rerank/cluster_prior.py`, `fusion.py`   |
| 3.4 Qwen 基线      | `src/m3sum/stage2_rerank/baselines/qwen3_vl_rerank.py`    |
| 3.5 Stage-3 生成   | `src/m3sum/stage3_generation/generators.py`               |
| 主配置              | `src/configs/trial_31.yaml`                               |


