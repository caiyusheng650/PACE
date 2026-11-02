"""Explicit legacy ID conversion, shared metadata Q-matrix, train-only pools."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch


def codes(value):
    if isinstance(value,str):
        value=ast.literal_eval(value)
    if isinstance(value,(int,float)):
        value=[value]
    return [int(k)-1 for k in value]


class Data:
    def __init__(self,repo,train_id_base=0,eval_id_base=1):
        self.repo=Path(repo)
        self.students,self.items,self.concepts=map(int,(self.repo/'config.txt').read_text().splitlines()[1].split(','))
        self.hashes={}
        self.rows={}
        paths={'train':self.repo/'data/neg_exer_ours.json',
               'validation':self.repo/'data/Eedi/val_set.json',
               'test':self.repo/'data/Eedi/test_set.json'}
        for split,path in paths.items():
            payload=path.read_bytes();self.hashes[split]=hashlib.sha256(payload).hexdigest()
            raw=json.loads(payload)
            if split=='train':
                rows=[(int(x['user_id'])-train_id_base,int(x['exer_id'])-train_id_base,float(x['score']),codes(x['knowledge_code'])) for x in raw]
            else:
                rows=[(int(s['user_id'])-eval_id_base,int(x['exer_id'])-eval_id_base,float(x['score']),codes(x['knowledge_code'])) for s in raw for x in s['logs']]
            for u,e,y,ks in rows:
                if not(0<=u<self.students and 0<=e<self.items and y in (0,1) and ks and min(ks)>=0 and max(ks)<self.concepts):
                    raise ValueError(f'{split}: invalid IDs/response/Q row; inspect explicit index bases')
            self.rows[split]=rows
        self.counts=Counter(str(u) for u,_,_,_ in self.rows['train'])
        for split in ['validation','test']:
            for u,_,_,_ in self.rows[split]:
                self.counts.setdefault(str(u),0)
        self.observed=defaultdict(set)
        for u,e,_,_ in self.rows['train']:
            self.observed[u].add(e)
        # Q is public exercise metadata. No validation/test responses enter training.
        self.q=torch.zeros(self.items,self.concepts)
        for rows in self.rows.values():
            for _,e,_,ks in rows:
                self.q[e,ks]=1
        self.by_concept=[set(torch.where(self.q[:,k]>0)[0].tolist()) for k in range(self.concepts)]
        self.valid_items=set(torch.where(self.q.sum(1)>0)[0].tolist())
        pairs={s:{(u,e) for u,e,_,_ in rows} for s,rows in self.rows.items()}
        self.overlap={a+'__'+b:len(pairs[a]&pairs[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
        self.within_split_duplicates={s:len(rows)-len(pairs[s]) for s,rows in self.rows.items()}

    def pool(self,user,item,rng,limit=100):
        allowed=set()
        for k in torch.where(self.q[item]>0)[0].tolist():
            allowed.update(self.by_concept[k])
        allowed.difference_update(self.observed[user])
        if not allowed:
            allowed=self.valid_items-self.observed[user]
        if not allowed:
            raise ValueError('No unobserved candidate for student; cannot substitute an observed response')
        ordered=sorted(allowed)
        return rng.sample(ordered,limit) if len(ordered)>=limit else rng.choices(ordered,k=limit)

    def batch(self,rows,device):
        return (torch.tensor([r[0] for r in rows],device=device),
                torch.tensor([r[1] for r in rows],device=device),
                torch.tensor([r[2] for r in rows],dtype=torch.float32,device=device))

