# 公开量化作者与可迁移内容

核查于2026-11-16北京时间。账户身份与代码可见不等于已验证的投资收益；没有移植公开项目的业绩数字，也没有把期货/美股研究直接称作A股有效策略。仓库revision和阅读文件SHA256见reports/v9/practitioner_sources.json。仅独立实现方法，不复制或执行外部仓库代码。

| 作者 | 身份依据 | 公开GitHub与本轮采用/不采用内容 |
|---|---|---|
| Robert Carver | [本人履历](https://www.systematicmoney.org/bio)：曾管理AHL固定收益组合 | [个人账户](https://github.com/robcarver17)；系统现迁至[pst-group/pysystemtrade](https://github.com/pst-group/pysystemtrade)。阅读预测缩放与仓位缓冲代码，采用预测/仓位分离及降低不必要换手的思想；不复制其GPL代码、期货合约或杠杆系统。本轮slow_buffer为既有排名缓冲的慢频对照，不宣称复刻其仓位带算法。 |
| Artur Sepp | [本人网站](https://artursepp.com/)与[公开账户](https://github.com/ArturSepp)列出量化研究及从业背景 | [QuantInvestStrats](https://github.com/ArturSepp/QuantInvestStrats)：核查风险与Sharpe分析文档，采用分离总收益/风险/基准归因、明确采样频率和区间。其工具不是可直接购买的α信号；本轮没有引入复杂波动率衍生品或声称复现其收益。 |
| Stefan Jansen | [作者介绍](https://www.ml4trading.io/about/)；属于技术作者与咨询背景，不将其描述为已验证业绩的基金经理 | [machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)：核查公司特征案例中价值/质量/动量、月频标签、分期验证与成本可行性。采用经济量纲修正、多期限消融和明确成本约束；不复用其美股数据、成本数值或长短组合结果。 |

额外核查[Bailey与López de Prado的DSR论文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)及[作者软件页面](https://www.quantresearch.org/Software.htm)。这用于识别重复选优风险；本轮保留试验账本与条件区间，没有因无法确定整个研究历史的有效独立试验数而报一个虚假的“已校正显著性”。没有找到并采用可独立认证的作者GitHub账户，故不将第三方复刻当作其本人代码。

## 对本项目的具体判断

V8的短窗口校准与原6个月选模相互挤占；V9以旧预测逐期成熟后再校准，增加信息利用但严格保留预测时点。EPS/BPS原值混淆每股规模与估值；V9先除同日原始收盘价格再排名，年度ROE只取已披露年度，TTM EPS仅是股份口径未认证的代理。H5弱优势未必覆盖费用；V9将训练/校准/风险的H20与慢频持仓联系起来，同时保留H5主指标及周频公平对照。

上述实现是待检验假设，不根据作者资历提前宣称有效。
