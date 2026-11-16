# 冻结预测时钟实验复核

在项目环境（pip install -e '.[v4]' pytest）中：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m alphacontextlab.v10_clock.audit
```

不联网、不拟合、不读pickle。逐文件核验V9输入哈希，重建两阶段108条费用路径、组预算、现金、持仓及基金净额，独立核验费用和资金守恒。phase0必须重现V9全部对照路径。

重新运行实验应检出provenance.json记载的干净源提交49106195e1f1af97e1be1e8ec58f5af0b1f6e2ab，使用新的输出目录：

```bash
python -m alphacontextlab.v10_clock.study development --outputs outputs/v10_clock_repeat/development
python -m alphacontextlab.v10_clock.study evaluation --outputs outputs/v10_clock_repeat/evaluation
```

主方案已在该源提交公开固定，不需要开发选优，不得再选择某个相位。输入全部来自已发布V9结果，不能将任何区间重新称为盲测。

本地原文核验额外需要pdftotext。按已发布清单恢复16份原文，严格核对SHA256；不会重查后重新选择文档，也不把本次获取时间伪称历史first_seen：

```bash
python -m alphacontextlab.v10_clock.catalogue --materialize
python -m alphacontextlab.v10_clock.audit --local-filings
```

两份年报的六个手工字段核验与16份归档的文档哈希检查分开报告。仅验证原文当前获取字节与日期证据，不认证历史字节身份/完整修订链。原文不收入公开仓库，公告元数据快照独立保留。本轮完全未使用新财报字段训练模型。发布后的清单不可直接覆盖，新增采集必须使用新版本。
