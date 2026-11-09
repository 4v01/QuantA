# V2 复现指南

先审查报告，再选择参数重放或从头拟合。发布仓库不含第三方原始行情或本地参数；本地复现快照提供缓存及参数，不能自动上传为公开数据集。项目MIT许可不覆盖行情。

## 公开算术与来源审计

```bash
git clone https://github.com/4v01/QuantA.git
cd QuantA
python -m pip install -e . pytest
python -m pytest -q
python -m alphacontextlab.v2.audit
```

审计不访问行情或模型：读取公开全部日收益，重算888行统计；核对35个最终拟合、85个候选、训练标签边界、因子预算、验证冠军与确认区间。它不能认证行情源真实性或替代独立重训。

## 本地快照重放

把复现快照中的data/与outputs/合并到仓库，保持data/quotes.csv为原V1/开发版本，data/v2/为确认版本。两者不能互换，程序检查指纹。

```bash
python -m alphacontextlab.v2.audit --local-replay
python -m alphacontextlab.v2.audit --retrain-one
```

前者检查70个验证/最终参数文件哈希并重放预测与验证效用；后者重新执行固定2021/evolved_lstm/seed17的因子选择、尺度拟合和训练，校验参数逐位一致。后者是额外一次工程复训，不进入模型选择或35个主实验计数。只加载本项目哈希匹配的本地pickle，不能把任意外部pickle当数据打开。

## 从头复训冻结源版本

为避免报告/封存提交改变训练来源，源代码使用独立detached工作树；数据和输出按.gitignore忽略。两个数据版本应从本地快照复制到工作树data/和data/v2/，否则需要重新下载并核对数据版本。

```bash
git worktree add --detach ../QuantA-v2-frozen 4bee73444c84eb900bc913b99d5da999d3eb0075
cd ../QuantA-v2-frozen
export PYTHONPATH=src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# 此处把归档data/放入该工作树；不覆盖原归档outputs/。
python -m alphacontextlab.v2.study development --data data --outputs outputs/v2_repeat
# 先检查outputs/v2_repeat/champion.json；新研究应从另一个工作树公开封存。
python -m alphacontextlab.v2.study fetch-confirmation --data data/v2 --outputs outputs/v2_repeat
python -m alphacontextlab.v2.study confirmation --data data/v2 --outputs outputs/v2_repeat
```

fetch-confirmation优先使用给定目录缓存，缺少缓存时才下载。重放已经见过的2025不再构成新的独立确认。源程序拒绝脏工作树、不同来源覆写、未封存下载和不完整报告。

拟合完成后，在主工作树用当前审计/报告工具读取新输出并写入新报告目录，保留原reports/v2不变。示意：

```bash
cd ../QuantA
python -m alphacontextlab.v2.report --outputs ../QuantA-v2-frozen/outputs/v2_repeat --reports reports/v2_repeat
cp ../QuantA-v2-frozen/outputs/v2_repeat/champion.json reports/v2_repeat/champion_pre_confirmation.json
python -m alphacontextlab.v2.audit --reports reports/v2_repeat
```

这里复制封存文件仅用于已有实验复现；新研究必须在下载未见时期前完成真实公开封存，不能把事后复制当作事前证据。原研究封存提交与文件获取顺序见reports/v2/acquisition_audit.json。
