# X机制与舆情：本轮做了什么，还缺什么

2026-11-10（北京时间）核对原始仓库。区别两个版本：

- [旧Twitter Home Mixer](https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md)：候选生成、特征补充、排序、过滤/多样性。V4已冻结的x_rerank是这一结构的独立极小金融适配：同LightGBM预测，加固定波动/流动性秩项，不声称移植原模型。
- [当前X官方xai-org/x-algorithm](https://github.com/xai-org/x-algorithm/tree/660cd7407c5726ec8d3aef27c32e1377b1c7e480)，核对时main提交`660cd7407c5726ec8d3aef27c32e1377b1c7e480`；[Phoenix文档](https://github.com/xai-org/x-algorithm/blob/660cd7407c5726ec8d3aef27c32e1377b1c7e480/phoenix/README.md)：Transformer用于召回与互动排序，生产数据等基础设施不随导出提供。该版本仅完成文档/接口研究，本轮未部署或复现Phoenix。当前小型启发臂不能代表新版X效果。

互动概率并不等于未来收益。直接拿社交排序模型套股票池缺少训练目标与行为数据，不能据其架构宣称会提高夏普。

下一轮可检验的适配是证券/新闻候选→可得性与去重过滤→收益、下行风险、换手成本分头预测→受约束组合。先冻结对照，再看同日期增量IC；本轮x_rerank没有通过门槛，不因最新文档另选参数覆盖已有结果。

## 舆情管道已实际运行

AKShare `stock_news_em('600000')`实际采集10条新闻，保存标题/内容、原链接、发布时间、首次观察时间与内容哈希。新闻正文留在本地，公开采集元数据与指纹。决策可用时刻为`max(published_at, first_seen_at)`，不把早发布日期当作我们当时已拥有档案的证明。

本轮采集发生在研究截止日之后，所以未加入历史训练。简明中文词典计数只是情绪代理，没有情绪标注准确率或收益提升证据。未来要按日持续采集，并处理转载去重、实体映射、修订时间、来源覆盖和标签抽样核验，才能做前向增量验证。没有开通后台服务，也没有向X/其他平台发消息。
