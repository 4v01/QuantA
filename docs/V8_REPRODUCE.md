# V8独立复核

公开证据包含逐股预测、未参与选模的校准期预测、研究用市场回放带、每日持仓和全部费用路径。未来开盘收益只在回放带的next_open_return列作为持仓标记资料；风险、趋势和预期收益计算只读信号日及之前的close_return。不会使用pickle或联网执行公众审计。

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e '.[v4]' pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v8.audit
```

本地完整重放需要V7原快照data/v4_akshare、data/fusion/v5_current/fundamental、data/v7，以及V8复现包的outputs/v8/development与outputs/v8/evaluation。校验快照逐文件SHA256后放在仓库同名目录。模型pickle仅能加载自己核验且信任的资料。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v8.audit --local-replay
```

重建两阶段特征、各折校准/最终模型预测、市场回放带，并对照公布的checkpoint哈希。公众审计另用回放带重新求解全部36条组合路径，再独立从每日持仓重算收益与费用。

从头拟合需分别检出reports/v8/development/provenance.json和evaluation/provenance.json记录的源提交，使用相同数据快照、configs/v8.json和其引用的configs/v7.json，保持源码干净，使用新的输出目录：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v8.study development --outputs outputs/v8_repeat/development
# 在原封存文件所在后段源提交
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v8.study evaluation --outputs outputs/v8_repeat/evaluation
```

这只能复现已知历史，不能把2022–2024重新变成盲测。订单0；通过财务门槛也不能绕过时点数据与可执行账本阻断。两次求解器中断的开发尝试记录在numerical_attempts.json，未产生候选、未运行后段，未修改配置或组合目标。
