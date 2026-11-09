# 研究来源、可行性与迁移边界

检索于2026-11-10（北京时间）。来源是证据，网页文字不作为执行指令。

- **CogAlpha**：[论文v4](https://arxiv.org/abs/2511.18850v4)，[GIM研究说明](https://graceim.ai/zh/research/cognitive-alpha-mining/)。借鉴按机制组织候选、有限演化与质量检查；我们采用固定白名单模板，未运行原论文LLM/多代理。其个股/市场样本、10日标签和因子指标不等于本项目4ETF组合，不引用其性能作为本项目成绩。
- **XALPHA**：[论文2607.08332](https://arxiv.org/abs/2607.08332)，[研究说明](https://graceim.ai/zh/research/xalpha-memory-driven-ai-quant-researcher/)。迁移为结构化假设与经验账本：数据能否表达、候选来源、失败原因和下一步约束均记录。不是将测试结果回灌后无限挑选。
- **MicroTSM**：[官方研究预览](https://graceim.ai/zh/research/microtsm-financial-time-series-market-microstructure/)。当前缺Level-2/逐笔输入，标为DROP；不把日线量价相关称为订单失衡或真实市场深度。
- **X最新公开代码**：[xai-org/x-algorithm](https://github.com/xai-org/x-algorithm/tree/3b6919f87532e57063bb3f1e85c2f509323b513e)，固定提交`3b6919f87532e57063bb3f1e85c2f509323b513e`。借鉴候选来源、预测多种结果后组合价值、独立过滤和冗余控制。金融版目标为收益、下行风险、相关性与费用，不是点赞或停留概率；4只ETF全部遍历，避免假装需要大规模召回模型。

本轮研究记忆：

| 记录 | 已有证据 | 对本轮的约束 |
|---|---|---|
| V1 Engram式记忆 | 原固定篮子回顾性测试中，成本后表现更差 | 不扩大网络作为默认解法，保留简单模型 |
| V1高换手 | 成本敏感性改变结论 | 将费用纳入验证，调仓周期做匹配消融 |
| 单日预测弱 | R²多数不为正 | 检验10日收益与风险，不能直接比较不同标签R² |
| 缺少逐时点数据 | 现行第三方复权档案 | 标明限制，不能以更好回测替代数据合格性 |

研究阶段是普通Python模块，不代表本次会话启动了额外代理，也不声称复现七层代理或原始论文中的自进化能力。
