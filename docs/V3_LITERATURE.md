# V3 文献检索与取舍

2026-11-10（北京时间）检索。优先选能直接针对当前估计误差、交易成本和回测选择问题的方法；不把论文里的其他市场成绩当作本项目指标。

| 文献与原始来源 | 与当前问题的关系 | 本轮实现及边界 |
|---|---|---|
| Boyd等，2017，[Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html) | 收益预测与风险、费用、持仓约束应统一考虑 | 实现长仓无杠杆单期凸二次规划SPO；不是完整MPC多期预测器，也不声称改进预测本身 |
| Gârleanu、Pedersen，2013，[Dynamic Trading with Predictable Returns and Transaction Costs](https://www.aqr.com/Insights/Research/Journal-Article/Dynamic-Trading-With-Predictable-Returns-and-Transactions-Costs) | 高频全量换仓可能浪费可预测收益 | 对照向目标移动一半；没有估计原文信号衰减，也不把固定0.5称为论文最优闭式策略 |
| Ledoit、Wolf，2004，[A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://doi.org/10.1016/S0047-259X(03)00096-4) | 样本协方差不稳定会放大组合权重误差 | sklearn LedoitWolf自动收缩，对照同窗口样本协方差；只有4个资产，大维理论不保证此例必有增益 |
| Moreira、Muir，2017，[Volatility-Managed Portfolios，作者预印本](https://law.yale.edu/sites/default/files/area/workshop/leo/leo17_moreira.pdf) | 高波动时是否应该少持仓 | 固定常数除以前60日估计方差的缩放，并限制不加杠杆；不是原文的完整因子/月频/归一化实验 |
| Cederburg等，2020，[On the Performance of Volatility-Managed Portfolios](https://www.sciencedirect.com/science/article/pii/S0304405X2030132X) | 方差管理的样本外优势可能不稳定 | 保留未缩放对照，不能假设缩放一定改善 |
| DeMiguel、Garlappi、Uppal，2009，[Optimal versus Naive Diversification](https://doi.org/10.1093/rfs/hhm075) | 估计误差可能令复杂组合不如1/N | 等权每周再平衡为强参照之一，使用同一现金/股数/费用账本 |
| Bailey、López de Prado，2014，[The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | 多次尝试和非正态会抬高被选中结果 | 公开试验数。有效独立搜索次数及序列独立假设无法确认，不输出看似精确的DSR数值 |
| White，2000，[A Reality Check for Data Snooping](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf) | 应考虑整组候选而非只看胜者 | 对57个验证候选做中心化、非学生化、20日循环区块max-mean诊断；不是Hansen SPA或对全部既往研究的全局修正 |

反面文献也被保留：波动管理和收缩是否有效由本轮匹配实验决定。上述方法均为公开思路的独立、小规模实现；没有调用付费服务、下载LLM或启动额外研究代理。

## 先修复数据与成交边界

原始价格与明确现金分红/拆分事件逐日构造总回报输入，未来事件不会重写此前特征。原始报价历史版本和分红公告精确可得时刻仍未认证，不能称为完整PIT数据。

510500拆分采用公告中的精确值与有效交易日：2015-04-15起0.28032483，份额小数舍去；2022-08-29起1.14539，小数上进位。后者网页显示的2022-08-26是日终登记日期，不能提前到当日开盘应用。

原始证据：[2015结果及复牌公告](https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf)、[2022拆分结果公告](https://pdf.dfcfw.com/pdf/H2_AN202208291577764682_1.pdf)。分红的登记、除息、支付日分别从四只基金的分红表缓存并保留哈希，不把金额在登记日提前加到NAV。
