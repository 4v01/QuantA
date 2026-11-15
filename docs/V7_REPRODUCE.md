# V7复现与独立审查

## 公开算术审计

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e '.[v4]' pytest
python -m pytest -q
python -m alphacontextlab.v7.audit
```

无需新闻/报价、不读pickle，重算两阶段所有方法的逐日IC、校准/选择规则、封存候选与财务门槛，以及90条费用路径算术。后段候选必须等于封存候选，不能因外段有更高点估计而替换。

## 受信本地快照重放

所需目录：data/v4_akshare、data/fusion/v5_current/fundamental、data/v7及outputs/v7。按归档清单逐文件验证，再放到源码仓库同名目录。

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m alphacontextlab.v7.audit --local-replay
```

检查70组checkpoint（210个最终组模型）的哈希，重建原特征与全部预测及90条收益筛查。仅使用本项目受信pickle；这不是对历史数据版本或真实成交的认证。

## 从冻结源提交重新拟合

开发源：2697579c8927fd75af65272a9be1e3fdb40dcfea。
候选封存及后段拟合源：ffc840ae2b94dca4f734ccb814216d06295f361a。
报告HEAD不是模型来源。分别建立独立worktree，放入完全相同的data缓存，保持源码clean，使用新输出目录：

```bash
# 开发源工作树
python -m alphacontextlab.v7.study development --outputs outputs/v7_repeat/development
# 封存源工作树（包含原封存文件）
python -m alphacontextlab.v7.study holdout --outputs outputs/v7_repeat/holdout
```

同一数据和配置下开发候选应为long_history。独立新研究需要重新公开封存，而复训已知2022–2024不会把它变成新盲测。精确环境在reports/v7/execution_environment.json；requirements-v4-lock.txt包含原环境依赖，CPU torch必要时用其官方CPU索引安装。

## 新闻来源重建

2018–2021为原data/v6/news.csv.gz及其manifest，58股映射与财务逐表哈希保留。2022–2024仅取作者FinMultiTime固定版本的title：

https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting/resolve/72d4e97b9acb92651949b8efb22c55676ff633ff/text/hs300news_summary.zip

原ZIP的URL/字节数/SHA256在data/v7/manifest.json的archive_receipt里。下载到新位置校验后，准备器可重建新的新闻缓存。不能把summary补到缺title处，也不按file_date回填其它未经核验的新闻CSV。

新的行情/财务API下载必须作为新版本，不覆盖旧缓存。当前研究是出版/更新日重建，缺原始修订链和first_seen档案，不将其用于旧V5严格准入。

## 解释成本诊断

cost_attribution.json是相同持仓的毛收益归因；不是降低主费用的新候选。20bp主费用、buffer主政策和封存候选都不因诊断改动。Kaggle方法只是独立适配，未下载比赛私有数据、复现排行榜成绩或调用需要授权的标签服务。
