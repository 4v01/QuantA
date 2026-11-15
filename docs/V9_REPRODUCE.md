# V9复现与独立审查

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e '.[v4]' pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v9.audit
```

公众审计不联网、不加载pickle：重建所有滚动收益校准、主/次IC、R²、开发选优、封存门槛与108条组合路径，再从每日持仓独立重算收益、换手和费用。matched_control的主要预测以及每边10/20/30bp周频缓冲路径必须重现V7 long_history，确保不是换弱基线。

## 本地输入与模型重放

使用未改动的QuantA-v7-data.zip输入快照（data/v4_akshare、data/fusion/v5_current/fundamental、data/v7）及本轮outputs/v9/development和evaluation。先按归档清单校验每个文件；只加载自己核验且信任的模型pickle。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v9.audit --local-replay
```

重建经济因子、缺失/滞后诊断、市场回放带、全部60组checkpoint预测和两期限标签。原财务快照含修订，不能把这一步说成认证了原始披露版本或真实成交。

## 从头拟合

分别检出两阶段provenance.json的source_commit，使用相同输入和新输出目录，源码保持clean：

```bash
python -m alphacontextlab.v9.study development --outputs outputs/v9_repeat/development
# 后段源提交包含原公开封存和开发逐期预测，不能重选候选
python -m alphacontextlab.v9.study evaluation --outputs outputs/v9_repeat/evaluation
```

开发包括2019预热预测，选优和收益评价仅2020–2021。后段校准从已公布的开发预测继承历史，再按H20真实退出日逐期成熟。重跑已知历史不构成新盲测。首次开发缺预测现金回退的修复记录见development_attempt1_correction.json。

## 可选数据来源探查

source_probe.py仅验证潜在替代来源，不更改V9训练输入、不产生订单。单独安装baostock==0.9.4到隔离target目录后，以该目录加入PYTHONPATH运行，外层限制55秒。回执为data_source_probe.json。即使有pubDate或历史成分股API响应，也需核验原始公告与修订链，不能自动称为严格PIT。该包没有加入拟合环境或CI依赖。
