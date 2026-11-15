# Kaggle公开方法核查与A股日频改造

2026-11-16核查。Kaggle写作页部分无可读正文，已记录入口；实际方法说明以作者公开GitHub文字和代码为据。不将搜索摘要当完整解法、不将比赛分数当本项目夏普。

| 公开原始来源 | 核实到的机制 | 本轮独立适配 | 没有直接照搬 |
|---|---|---|---|
| [Jane Street作者solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/8598feb3ae28e470d1ed65661261f6f035a6e002/solution.md)，[Kaggle入口](https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution) | 时间CV及间隔检验；同时间市场均值、每标的滚动统计；辅助期限目标；标签可得后的更新 | 同日市场状态/交互、基本面H20目标、严格退出日purge和后续年份检查 | 未下载比赛数据/训练大GRU，未复制其多responder构造；本项目无法在未到期时获得未来标签，绝不提前在线更新 |
| [Ubiquant作者代码](https://github.com/pinouche/ubiquant-kaggle-competition/tree/2a275c7af0cda02d830c0facbac237d2ffa29568) | Notebook有同time_id截面秩/标准化、聚合统计、LightGBM与相关性有关目标 | 保留同日截面秩；增加受限市场上下文和小型树模型对照 | 不照搬匿名特征、未审定的随机切分/排行榜排名，不拿自报top1%作独立认证 |
| [Optiver团队公开仓库](https://github.com/cvaisnor/optiver_kaggle_comp/tree/7fc09e2678c73b8a780749c1ddf6916e80e92f32)，[官方竞赛说明](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview) | 收盘竞价相对价格目标、时间/市场特征与模型比较，仓库含LightGBM相关Notebook | 结合本项目的截面排序目标，新增一个小型LambdaRank价量分支 | 日线没有竞价盘口字段，不虚构bid/ask/auction imbalance。LambdaRank是本项目针对排名的独立改造，不声称是原冠军模型 |

仓库许可证字段均未提供单独许可，不复制代码到项目或执行外部脚本；仅阅读公开思路，独立实现。原作者比赛名次/增益属于其自报、协议不同，未用于本项目结论。源码提交与核查文件清单存研究来源JSON。

本轮将Kaggle方法与本项目原问题一起检验：单窗口翻向、尺度等幅放大弱组、历史过窄、慢财务目标不匹配。不是用“换比赛冠军模型”代替时间验证。两项Kaggle适配都有前置线性对照，全部结果保留。

## 新数据扩展的边界

[FinMultiTime作者论文](https://arxiv.org/abs/2506.05019)给出作者[数据入口](https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting)。本轮只下载HS300新闻ZIP（约57MB），固定revision72d4e97b9acb92651949b8efb22c55676ff633ff。仅取datetime/title，不用summary，58股票均有文件；2022–2024得43,231条去重非空标题。source title字段也不等于已认证的原发布版本，因此不作严格PIT声明。

另核查了此前GitHub原始新闻CSV：实际datetime在2024–2025、file_date却出现2020，且编码解码字段混杂、源档案说明不足。未用于本轮训练/标签或按file_date回填，以免引入错误历史。新数据源不会自动消除样本选择和公告修订风险。
