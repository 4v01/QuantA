# V4：复现与模拟准入

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e '.[v4]' pytest
python -m pytest -q
python -m alphacontextlab.v4.audit
```

可用`python -m alphacontextlab.v4.audit --local-replay`重放本项目受信本地快照的12组模型及全部18条曲线。

公开审计不取数，不读取pickle：用逐股票OOF预测/收益代理重算每日IC、全部区块区间、18条收益成本曲线、验证选择及准入。源提交和环境版本见reports/v4/provenance.json，27因子与全部60代码/门槛见configs/v4.json。

## 精确快照复训

使用本地QuantA-v4-repro.zip的data/v4_akshare，先核对REPRO_SNAPSHOT.json逐文件SHA256，不能拿新下载版本冒充旧数据。在报告provenance.source_commit建立独立干净工作树，将快照data目录复制过去：

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m alphacontextlab.v4.study run --outputs outputs/v4_repeat
python -m alphacontextlab.v4.study report --outputs outputs/v4_repeat --reports reports/v4_repeat
python -m alphacontextlab.v4.audit --reports reports/v4_repeat
```

新报告目录在运行结束后生成；再次正式run要求先归档/提交干净工作树。不覆盖已有outputs/reports。每个半年模型与GP公式保存，pickle仅用于自有受信文件，公开审计不需要加载。

## 获取新版本

```bash
python -m alphacontextlab.v4.study fetch --data data/v4_akshare_new
```

单一东方财富接口，无声换源/丢股不允许。失败文件记录failures.json，再运行同命令仅重取缺失文件；已存CSV/侧车哈希不符立即拒绝。旧数据保留。限流时等待，不绕过服务限制。第三方接口会修订历史；新manifest就是新实验数据版本。

## 舆情前向采集

```python
from alphacontextlab.v4.sentiment import collect, available_records
path = collect('600000', 'data/v4_news')
```

只用published_at和first_seen_at均早于决策时间的新闻。当前采集不是完整历史档案，未接入本轮历史回测。采集命令不会后台常驻或外发消息。

## 模拟准入检查

```bash
python -m alphacontextlab.v4.paper
```

未达门槛退出码2，留下本地admission JSON，订单数量0。该命令是准入记录器，不是经纪商接口/常驻模拟成交服务。不能通过修改status绕过股票池/现金账本认证。真正进入未来纸面交易前，要先补齐这些结构条件并通过信号/净夏普门槛；不将本轮历史组合筛查冒充模拟盘成绩。
