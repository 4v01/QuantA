# GitHub来源核查：代码、数据、时间与本项目瓶颈

2026-11-10核查，8个仓库的实际提交、许可证字段与归档状态存data/v6/github_sources.json，发布报告时复制公开。不是拿仓库Star或外部论文的夏普当本项目benchmark。

| 原始仓库 | 实际内容与覆盖 | 解决什么 | 本轮取舍 |
|---|---|---|---|
| [BigRoddy/CMIN-Dataset](https://github.com/BigRoddy/CMIN-Dataset) | ACL2023作者实际数据，CMIN-CN新闻/价格2018–2021；MIT仓库 | 缺历史公司新闻 | 采用无表头原始三字段XLSX，核对58个共同股票；433,482条去重标题，逐文件哈希。只用日期/标题/来源，不取其收益标签/价格做输入 |
| [rootSunc/CNEquity](https://github.com/rootSunc/CNEquity) | Apache-2.0采集/数据湖框架，有溯源/历史回补机制；仓库非完整历史数据库 | 未来历史股票池、公司行动、财务采集组织 | 核对文档/目录，借鉴数据契约；不因README写PIT就认证本项目的原始历史版本，不启动全市场数小时任务 |
| [microsoft/qlib](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/pit) | MIT PIT采集/格式工具，季度财务来源BaoStock；官方明确数据可能不完善 | 财报发布期与修订存储 | 核对collector文档，采用明确可得日期的设计；本轮使用已归档财务的max公告/更新日期+1天，并仍标记未认证历史版本，不假装得到官方原始报表链 |
| [AI4Finance-Foundation/FinNLP](https://github.com/AI4Finance-Foundation/FinNLP) | MIT采集/数据来源索引；Astock、StockNet是子模块 | 找历史新闻/社交数据 | 实际核对.gitmodules，追到原作者仓库。没有把美股87股的2014–2015推文当A股2020年信息 |
| [JinanZou/Astock](https://github.com/JinanZou/Astock) | FinNLP2022作者新闻数据与模型；另含标签、价格变化、预计算文本；未见单独许可证字段 | 备选公司新闻 | 读原文件树/论文信息，主试验优先选择明确MIT、原始字段更易审查的CMIN。不复用其全股票可做空/发布时成交的回测成绩 |
| [Polymarket/py-clob-client](https://github.com/Polymarket/py-clob-client) | 公开CLOB客户端，仓库已归档；不附A股实体映射与所需早年赔率档案 | API代码而非历史映射瓶颈 | 记录版本/归档状态，保留V5的只读现时采集，不为2020补造预测市场信号 |
| [stac-utils/pystac-client](https://github.com/stac-utils/pystac-client) | STAC检索客户端，不附厂址/像素经济活动训练标签 | 卫星检索工具 | 保留目录接入，未用云量或场景数冒充工厂产能。完整经济测量仍待后续映射与处理 |
| [liuqi6776/news_stock_research](https://github.com/liuqi6776/news_stock_research) | 实际文件树大量CSV为LFS指针，部分情绪/股票匹配是后处理结果；未见单独许可证字段 | 另一历史新闻候选 | 只核查内容与字段，不把作者后验LLM情绪、收益关联或报告成绩当我们独立模型输入/结果 |

CMIN数据集原文：[Luo et al., ACL2023](https://aclanthology.org/2023.acl-long.679/)。只将数据用于当前可复现研究，不声称复现论文CMIN网络或其结果。MIT仓库许可不自动消除原新闻出版者权利，原文与财务/报价快照仅本地保存，公开派生指标与指纹。

## 当前能替代什么

新的V6配置将另类30%明确改为历史A股新闻100%，基本面30%和价量40%不变；这是新声明的来源替代实验，不是V5缺源后自动重分配。旧严格配置与旧结果不改。

CMIN不能补齐2022–2026，现时财务表也不能证明每次历史修订。因此首轮只在共同2018–2021资料内做2020诊断、2021复核；结果仅属于按已记录出版/更新日重建的回顾性研究。严格first_seen规则未被回填或关闭，模拟盘仍不准入。
