"""Prospective-only AKShare news cache: publication AND first-seen availability."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import pandas as pd
from ..study import write_json
from .data import utcnow

POSITIVE=('增长','增持','回购','盈利','突破','中标','利好')
NEGATIVE=('下滑','减持','亏损','处罚','违约','诉讼','风险')


def collect(symbol,root):
    import akshare as ak
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seen=utcnow();d=ak.stock_news_em(symbol=symbol)
    required={'新闻标题','新闻内容','发布时间'}
    if not required<=set(d):raise ValueError('News schema changed')
    rows=[]
    for _,row in d.iterrows():
        title=str(row['新闻标题']);text=str(row['新闻内容']);stamp=pd.Timestamp(row['发布时间'])
        if stamp.tzinfo is None:stamp=stamp.tz_localize('Asia/Shanghai')
        published=stamp.tz_convert('UTC').isoformat();content=title+'\n'+text
        rows.append({'symbol':symbol,'published_at':published,'first_seen_at':seen,
                     'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'title':title,'content':text,
                     'url':str(row.get('新闻链接','')),'source':str(row.get('文章来源','')),
                     'sentiment_proxy':sum(content.count(x) for x in POSITIVE)-sum(content.count(x) for x in NEGATIVE)})
    p=root/(symbol+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    write_json(p,{'api':'akshare.stock_news_em','fetched_at':seen,'records':rows,
                  'limitations':['Small finance word-count proxy, not trained sentiment ground truth',
                                 'Never backdate first_seen_at; this snapshot is not a historic news archive']})
    return p


def available_records(records,decision_time):
    cutoff=pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:raise ValueError('Timezone-aware decision time required')
    return [r for r in records if max(pd.Timestamp(r['published_at']),pd.Timestamp(r['first_seen_at']))<=cutoff]
