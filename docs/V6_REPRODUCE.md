# V6复现与审查

## 无需原始新闻/报价的公开审计

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e '.[v4]' pytest
python -m pytest -q
python -m alphacontextlab.v6.audit
```

重算首轮+三个随访case的逐日IC、区块区间、固定融合权重、选择/方向、60条费用曲线；核对随访三臂的价量和基本面预测逐行一致。source config来自记录的git提交，而非将报告HEAD当训练来源。公开审计不读pickle，不取数。

## 自有受信快照重放

精确快照目录：data/v4_akshare、data/fusion/v5_current/fundamental、data/v6和outputs/v6。按REPRO_SNAPSHOT.json逐文件校验后放在仓库同名目录：

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m alphacontextlab.v6.audit --local-replay
```

加载8组哈希匹配的自有模型（24个最终组模型），从原始版本重建特征、全部组/融合预测及60条收益筛查。不要用不受信pickle。这个检查证明同环境计算可复现，不认证历史数据版本或真实成交。

## 从冻结源码重新拟合

首次源为7b4ee5ec33bbeb76a3a643a61a317eca0891063f；改进方向封存及随访源为8c50a43f2e3b97bda49f7305cf50385900427c55。分别建立独立git worktree，放入相同版本的data缓存。模型源码目录必须clean。归档输出保留，不覆盖：

```bash
# 首次源工作树中
python -m alphacontextlab.v6.study pilot --outputs outputs/v6/pilot_repeat
# 随访源工作树中（已经含公开首轮证据和改进计划）
python -m alphacontextlab.v6.study followup --variant baseline --outputs outputs/v6/followup_baseline_repeat
python -m alphacontextlab.v6.study followup --variant entity_filter --outputs outputs/v6/followup_entity_filter_repeat
python -m alphacontextlab.v6.study followup --variant entity_tfidf --outputs outputs/v6/followup_entity_tfidf_repeat
```

用当前报告工具分别publish到新的报告目录。审计函数audit_case/replay_case可指定单个case与相应outputs路径；默认v6.audit审查已发布的四个正式路径。重拟合已知资料不会让2021重新变成未见样本。

精确依赖见requirements-v4-lock.txt（本轮无新增依赖）；必要时使用`--extra-index-url https://download.pytorch.org/whl/cpu`安装其中CPU torch。实验要求两线程，环境另存reports/v6/execution_environment.json。

## 从GitHub重建新闻缓存

```bash
git clone --filter=blob:none --no-checkout https://github.com/BigRoddy/CMIN-Dataset.git ../CMIN-Dataset
git -C ../CMIN-Dataset sparse-checkout set --no-cone '/CMIN-CN/news/raw/' '/CMIN-CN/price/raw/' '/README.md' '/LICENSE'
git -C ../CMIN-Dataset checkout 3f571a53b81d8f3ea7b3afb9877bdbdbeae54c97
```

当前data.prepare可显式传source与新的root，仅读数据不执行第三方模型代码：

```python
from alphacontextlab.v6.data import prepare
prepare(source='../CMIN-Dataset', root='data/v6_repeat')
```

财务和行情必须使用原快照；新API下载是新版本，不宣称精确复现旧实验。CMIN源文件、归一缓存、价格manifest、财务逐表指纹全部保留。研究模式按记录出版/更新日重建，不会修改V5严格first_seen特征库或解除模拟准入。
