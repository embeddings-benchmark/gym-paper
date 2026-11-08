"""Judge the MTEB Arena's human retrieval votes (for 9_arena.py). One document per side, as people saw it. Pointwise:
grade each side. Pairwise: choice in both presentation orders. Talks to served decision models (/v1/decide), as the
gym's DecisionModel does; resumable.

    python arena_judge.py ../data/arena/<judge>.jsonl http://host:port [http://host2:port ...]
"""
import json,glob,sys,math,threading,time,itertools,collections,random,requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from huggingface_hub import snapshot_download
D=glob.glob(snapshot_download('mteb/arena-results',repo_type='dataset',allow_patterns=['data/retrieval_battle-*.jsonl'])+'/data/retrieval_battle-*.jsonl')
V={'leftvote':'A','rightvote':'B','tievote':'tie','bothbadvote':'both_bad'}
rows=[];skip=collections.Counter();seen=set()
for f in D:
    for l in open(f,errors='replace'):
        l=l.strip()
        if not l: continue
        r=json.loads(l); k=r.get('type')
        if k not in V: skip[k]+=1; continue
        da=[str(x) for i in r.get('0_output') or [] for x in i[1:]]; db=[str(x) for i in r.get('1_output') or [] for x in i[1:]]
        if not da or not db: skip['empty']+=1; continue
        rows.append(dict(id=f"{r.get('0_conv_id')}__{r.get('1_conv_id')}__{k}",q=str(r['0_prompt']),a=da[0],b=db[0],ma=r['0_model_name'],mb=r['1_model_name'],vote=V[k],corpus=str(r.get('0_corpus'))))
print('rows',len(rows),'skipped',dict(skip),'votes',collections.Counter(r['vote'] for r in rows),'corpora',collections.Counter(r['corpus'] for r in rows).most_common(6),'multi-doc sides',sum(0 for _ in []),flush=True)
OUT=sys.argv[1]; urls=sys.argv[2:]; it=itertools.cycle(urls); lk=threading.Lock()
def cut(t,n):
    if len(t)<=n: return t
    w=t[:n].rsplit(None,1); return (w[0] if len(w)>1 else t[:n])+'...'
GQ='How relevant is the document to the query? 0 = not relevant at all, 5 = perfectly relevant: it fully satisfies the information need.'
PQ='Which result list better satisfies the query, judging relevance, coverage of the information need and ranking quality?'; OP=['System A','System B','Tie']
def dec(body):
    for i in range(8):
        with lk: u=next(it)
        try:
            r=requests.post(u+'/v1/decide',json=body,headers={'Authorization':'Bearer gym'},timeout=900);r.raise_for_status();o=r.json();return dict(zip(o['options'],o['probabilities']))
        except Exception: time.sleep(3*(i+1))
    raise RuntimeError('decide failed')
out=open(OUT,'a'); done={}
try:
    for l in open(OUT):
        d=json.loads(l); done[d['id']]=d
except FileNotFoundError: pass
def one(r):
    if r['id'] in done: return done[r['id']]
    g=lambda t:sum(int(k)*v for k,v in dec({'kind':'score','state':f"Query: {r['q']}\nDocument: {cut(t,16000)}",'question':GQ}).items())
    ga,gb=g(r['a']),g(r['b'])
    st=lambda x,y:f"Query: {r['q']}\n\nSystem A results:\n  1. {cut(x,4000)}\n\nSystem B results:\n  1. {cut(y,4000)}"
    if r['a']==r['b']: pa=.5
    else:
        p1=dec({'kind':'choice','state':st(r['a'],r['b']),'question':PQ,'options':OP}); p2=dec({'kind':'choice','state':st(r['b'],r['a']),'question':PQ,'options':OP})
        pa=((p1['System A']+.5*p1['Tie'])+(p2['System B']+.5*p2['Tie']))/2
    d=dict(id=r['id'],vote=r['vote'],ga=ga,gb=gb,pa=pa,ma=r['ma'],mb=r['mb'],corpus=r['corpus'],same=r['a']==r['b'])
    with lk: out.write(json.dumps(d)+'\n'); out.flush()
    return d
t0=time.time()
with ThreadPoolExecutor(96) as p: R=list(p.map(one,rows))
print('judged',len(R),round(time.time()-t0),'s',flush=True)
