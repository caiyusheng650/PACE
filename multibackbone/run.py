"""Fresh, isolated multi-backbone runs; checkpoint/ESDC selection on selection users only."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch

from backbones import Backbone,PaceModel
from data import Data

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
import adop


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def predict(net,data,rows,device,batch):
    net.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            block=rows[start:start+batch];u,e,y=data.batch(block,device)
            p=net(u,e).cpu().tolist()
            out.extend({'sample_id':start+j,'user_id':str(r[0]),'item_id':r[1],'label':int(r[2]),'score':float(s)} for j,(r,s) in enumerate(zip(block,p)))
    return out


def train_one(args,data,manifest,seed,variant,root,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    backbone=Backbone(args.backbone,data.students,data.items,data.concepts,args.latent)
    net=PaceModel(backbone,data.q,variant,args.candidates).to(device)
    directory=root/(variant+'_'+str(seed));directory.mkdir(exist_ok=False)
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    selection=set(manifest['selection'])
    vrows=[r for r in data.rows['validation'] if str(r[0]) in selection]
    if not vrows:raise ValueError('Empty model-selection partition')
    rows=list(data.rows['train']);rng=random.Random(seed)
    if args.max_train:
        # Fixed subset across variants/seeds in bounded engineering pilots only.
        order=list(range(len(rows)));random.Random(args.split_seed).shuffle(order)
        rows=[rows[i] for i in order[:args.max_train]]
    best=-1.;history=[];started=time.time()
    # Align training RNGs after variant-specific initialization (the gate has extra parameters).
    torch.manual_seed(seed+991)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed+991)
    for epoch in range(args.epochs):
        rng.shuffle(rows);net.train();losses=[]
        for start in range(0,len(rows),args.batch_size):
            block=rows[start:start+args.batch_size];u,e,y=data.batch(block,device)
            pool=None
            if variant!='original':
                pool=torch.tensor([data.pool(r[0],r[1],rng,args.pool_size) for r in block],device=device)
            opt.zero_grad(set_to_none=True)
            loss,components=net.objective(u,e,y,pool)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();opt.step();net.backbone.clip();losses.append(float(loss.detach()))
            step=start//args.batch_size+1
            if step==1 or step%100==0 or start+args.batch_size>=len(rows):
                write(directory/'progress.json',{'phase':'training','epoch':epoch+1,'step':step,
                      'steps_per_epoch':(len(rows)+args.batch_size-1)//args.batch_size,
                      'last_loss':float(loss.detach()),'elapsed_seconds':time.time()-started})
        val=predict(net,data,vrows,device,args.eval_batch)
        auc=adop.auc(val)
        item={'epoch':epoch+1,'training_loss':sum(losses)/len(losses),'selection_auc':auc,'elapsed_seconds':time.time()-started}
        history.append(item);write(directory/'training_history.json',history)
        if auc>best:
            best=auc
            torch.save({'state_dict':net.state_dict(),'epoch':epoch+1,'variant':variant,'seed':seed,'backbone':args.backbone},directory/'best.pt')
        print(args.backbone,variant,seed,item,flush=True)
    state=torch.load(directory/'best.pt',map_location=device,weights_only=True)
    net.load_state_dict(state['state_dict'],strict=True)
    # Export full validation AFTER freezing the checkpoint. No test evaluation here.
    val=predict(net,data,data.rows['validation'],device,args.eval_batch)
    write(directory/'validation_predictions.json',val)
    return {'directory':str(directory),'variant':variant,'seed':seed,'epoch':state['epoch'],'net':net,'validation':val}


def select_committee(runs,manifest):
    users=set(manifest['selection']);indices=[i for i,r in enumerate(runs[0]['validation']) if r['user_id'] in users]
    for run in runs[1:]:
        assert [(x['user_id'],x['item_id'],x['label']) for x in run['validation']]==[(x['user_id'],x['item_id'],x['label']) for x in runs[0]['validation']]
    selected=[];current=np.zeros(len(indices));best=-1
    for _ in range(8):
        chosen=None;score=best
        for j,run in enumerate(runs):
            p=np.array([run['validation'][i]['score'] for i in indices])
            trial=(current*len(selected)+p)/(len(selected)+1)
            rows=[dict(run['validation'][i],score=float(s)) for i,s in zip(indices,trial)]
            value=adop.auc(rows)
            if value>score+1e-6:chosen=j;score=value
        if chosen is None:break
        p=np.array([runs[chosen]['validation'][i]['score'] for i in indices])
        current=(current*len(selected)+p)/(len(selected)+1);selected.append(chosen);best=score
    return selected


def average(rows):
    out=[dict(r) for r in rows[0]]
    for run in rows[1:]:
        assert [(r['user_id'],r['item_id'],r['label']) for r in run]==[(r['user_id'],r['item_id'],r['label']) for r in out]
    for i,r in enumerate(out):r['score']=sum(run[i]['score'] for run in rows)/len(rows)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backbone',choices=['ncdm','mirt','kancd','kscd','honcd'],required=True)
    p.add_argument('--variants',nargs='+',choices=['original','fixed','gate'],default=['original','fixed','gate'])
    p.add_argument('--seeds',nargs='+',type=int,default=[2024,42,7]);p.add_argument('--split-seed',type=int,default=2024)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--latent',type=int,default=32)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--eval-batch',type=int,default=256);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--candidates',type=int,default=20);p.add_argument('--pool-size',type=int,default=100)
    p.add_argument('--max-train',type=int,default=0);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--gpu-memory-gb',type=float,default=0,help='Limit this process allocator; 0 leaves the PyTorch default')
    p.add_argument('--train-id-base',type=int,choices=[0,1],default=0)
    p.add_argument('--eval-id-base',type=int,choices=[0,1],default=1)
    p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--allow-pair-overlap',action='store_true',help='Exploratory only; overlap is always recorded')
    args=p.parse_args()
    if args.epochs<1 or args.candidates<1 or args.pool_size<1: p.error('Positive epochs/candidate counts required')
    torch.set_num_threads(args.threads);device=torch.device(args.device)
    if device.type=='cuda' and args.gpu_memory_gb:
        total=torch.cuda.get_device_properties(device).total_memory
        fraction=args.gpu_memory_gb*1024**3/total
        if not 0<fraction<=1:p.error('GPU memory limit must be positive and no larger than device capacity')
        torch.cuda.set_per_process_memory_fraction(fraction,device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',vars(args))
    try:
        data=Data(args.repo,args.train_id_base,args.eval_id_base)
        source_hash={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
        write(out/'data_audit.json',{'hashes':data.hashes,'pair_overlap':data.overlap,'within_split_duplicates':data.within_split_duplicates,
                                   'students':data.students,'items':data.items,'concepts':data.concepts,'source_hashes':source_hash,
                                   'id_convention':'canonical zero-based student/item; explicit legacy conversion',
                                   'engineering_pilot_only':bool(args.max_train)})
        if any(data.overlap.values()) and not args.allow_pair_overlap:
            raise ValueError('Cross-split student/exercise overlap; inspect data_audit.json rather than silently permitting leakage')
        manifest=adop.split_students([r[0] for r in data.rows['validation']],args.split_seed)
        write(out/'validation_roles.json',manifest)
        runs=[train_one(args,data,manifest,seed,v,out,device) for v in args.variants for seed in args.seeds]
        pools=[r for r in runs if r['variant'] in {'fixed','gate'}]
        chosen=select_committee(pools,manifest) if pools else []
        descriptions=[{k:r[k] for k in ['directory','variant','seed','epoch']} for r in pools]
        write(out/'committee_locked.json',{'candidates':descriptions,'sequence':chosen,'test_used':False})
        evaluations=[(r['variant']+'_'+str(r['seed']),r['validation'],[r]) for r in runs]
        if chosen:evaluations.append(('pace_committee',average([pools[i]['validation'] for i in chosen]),[pools[i] for i in chosen]))
        locked=[]
        for name,val,members in evaluations:
            params=adop.fit_adop(val,dict(data.counts),manifest)
            write(out/(name+'_adop_locked.json'),params)
            locked.append((name,val,members,params))
        # Only after every model, committee and threshold has been frozen is test scoring allowed.
        if args.evaluate_test:
            cache={};results=[]
            for name,val,members,params in locked:
                predictions=[]
                for member in members:
                    key=member['directory']
                    if key not in cache:cache[key]=predict(member['net'],data,data.rows['test'],device,args.eval_batch)
                    predictions.append(cache[key])
                test=average(predictions)
                for row in test:data.counts.setdefault(row['user_id'],0)
                write(out/(name+'_test_predictions.json'),test)
                result={'name':name,'auc':adop.auc(test),'rmse':sum((r['score']-r['label'])**2 for r in test)**.5/len(test)**.5,
                        'acc_at_0_5':adop.accuracy(test,lambda r:.5),
                        'acc_global':adop.accuracy(test,lambda r:params['global_threshold']),
                        'acc_adop':adop.accuracy(test,lambda r:params['thresholds'][adop.group(data.counts[r['user_id']])])}
                results.append(result)
            write(out/'test_results.json',results)
        write(out/'status.json',{'status':'completed','test_evaluated':args.evaluate_test,'engineering_pilot_only':bool(args.max_train)})
    except Exception as e:
        write(out/'status.json',{'status':'failed','error':repr(e)})
        raise


if __name__=='__main__':main()

