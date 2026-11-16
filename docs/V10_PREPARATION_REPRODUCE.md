# 准备材料复核

普通审计不联网、不需要原始PDF或pickle：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python -m alphacontextlab.prep_v10.audit
```

重算已有V9逐股预测与持仓的决策日诊断，核验旧报告哈希、已使用历史范围和准备状态。审计通过表示“准备证据一致”，不表示已通过数据、收益或模拟盘门槛。

原始文档复核需要系统安装pdftotext及未改动的V7数据快照。文档约10MB，保留在忽略提交的data目录，不把原文全文纳入GitHub：

```bash
python -m alphacontextlab.prep_v10.source_pilot
python -m alphacontextlab.prep_v10.audit --local-documents
python -m alphacontextlab.prep_v10.build --output outputs/prep_v10_repeat
```

下载器核验固定SHA256；上游文件变化会失败，不覆盖旧版本。首次获取时间按本次实际下载记录，不能伪称为2021年first_seen。重复生成的获取时间可能不同；文档内容、原文页码、数值和决策诊断应一致。

准备构建器拒绝覆盖已有输出目录。没有训练或下单入口。收益语义、时点、历史成员、错峰时钟及袖套净额的合成检查在tests/test_prep_v10.py；真实股票现金账本仍未实现/认证。
