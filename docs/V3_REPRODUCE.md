# V3复现：先查公开算术，再重放现金账本

## 无需行情的公开审计

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e . pytest
python -m pytest -q
python -m alphacontextlab.v3.audit
```

检查源提交、配置/事件/报价版本、18个预测候选、342条验证路径、57候选的条件诊断、冠军封存与2026门槛，重算462行年度/拼接/敏感性统计；并检查V1/V2旧文件哈希未变。不需要加载外部pickle。

## 精确缓存重放

本地复现快照中data/v2为开发报价，data/v3_prices为2026检查报价，data/v3_corporate为事件缓存，outputs/v3为参数和全部账本。把它们放到仓库同名目录；不可互相替换数据版本。

```bash
python -m alphacontextlab.v3.audit --local-replay
```

该检查加载12个本项目哈希匹配的参数文件，重新生成114条所选预测器的验证账本，以及396条政策/参照/滑点/规模账本。只在自己的受信缓存使用pickle；这证明同环境可复现，不认证市场撮合假设。

## 从头复训

正式源提交为`a740f453764e317fc0b6a592d84ed0206b56a328`。代码提交、公开封存和后续报告提交不同，不能把报告HEAD冒充训练来源。用独立工作树固定源版本，数据/输出目录按.gitignore忽略：

```bash
git worktree add --detach ../QuantA-v3-frozen a740f453764e317fc0b6a592d84ed0206b56a328
cd ../QuantA-v3-frozen
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# 把归档data/放到这里；新输出避免覆盖原实验。
python -m alphacontextlab.v3.study development --outputs outputs/v3_repeat
# 先检查champion.json；新研究在另一工作树公开封存后再取新行情。
python -m alphacontextlab.v3.study fetch-confirmation --outputs outputs/v3_repeat
python -m alphacontextlab.v3.study confirmation --outputs outputs/v3_repeat
```

这是复训已有实验，不会让已经见过的2026重新成为未见样本。fetch-confirmation优先使用归档报价。若需要新下载，使用新缓存并标注新版本；禁止覆盖原始数据后宣称逐位复现。

从当前主工作树的报告工具读取新输出并写入新的reports/v3_repeat，再复制对应封存快照为champion_pre_confirmation.json，可做复现算术检查；不能把事后复制当新研究的事前公开证据。

## 获取当前公开数据（不是原数据版本）

主工作树提供`scripts/fetch_v3_events.py --root data/v3_corporate_new`，下载四张分红表和两份原始拆分公告后解析。未知拆分会停止，须重新核对主公告，不擅自套用四舍五入字段。

开发报价可以调用`alphacontextlab.v3.study.fetch_raw`，将配置end临时设为2025-12-31写入新目录；之后用该目录显式运行开发阶段。原始网络接口会修订历史，页面哈希也可能变化；新的报价/事件指纹只能作为新数据版本。数据来自第三方，不等于有完整PIT档案或任意再分发许可。
