import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from backbones import Backbone,PaceModel
from data import Data


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23);torch.set_num_threads(2)
        self.u=torch.tensor([0,1,2,0]);self.e=torch.tensor([1,2,3,4])
        self.q=torch.tensor([[1.,0.,1.],[0.,1.,0.],[1.,1.,0.],[0.,0.,1.]])

    def test_mirt_is_multidimensional_2pl(self):
        b=Backbone('mirt',4,7,3,latent=5)
        expected=torch.sigmoid((b.student(self.u)*F.softplus(b.item(self.e))).sum(-1)-b.discrimination(self.e).squeeze(-1))
        torch.testing.assert_close(b(self.u,self.e,self.q),expected)
        self.assertEqual(b.student.weight.shape,(4,5))

    def test_kancd_gmf_reference(self):
        b=Backbone('kancd',4,7,3,latent=5).eval()
        ability=torch.sigmoid(b.ability_map(b.student(self.u)[:,None,:]*b.knowledge)).squeeze(-1)
        diff=torch.sigmoid(b.difficulty_map(b.item(self.e)[:,None,:]*b.knowledge)).squeeze(-1)
        x=torch.sigmoid(b.discrimination(self.e))*(ability-diff)*self.q
        for layer in b.head[:2]:x=torch.tanh(F.linear(x,layer.weight.abs(),layer.bias))
        expected=torch.sigmoid(F.linear(x,b.head[2].weight.abs(),b.head[2].bias)).squeeze(-1)
        torch.testing.assert_close(b(self.u,self.e,self.q),expected)

    def test_kscd_reference_per_concept(self):
        b=Backbone('kscd',4,7,3,latent=5).eval()
        ability=(b.student(self.u)@b.knowledge.T).sigmoid()
        diff=(b.item(self.e)@b.knowledge.T).sigmoid()
        probabilities=[]
        for k in range(3):
            knowledge=b.knowledge[k].expand(4,-1)
            pref=torch.tanh(b.ks1(torch.cat([ability,knowledge],-1)))
            difficulty=torch.tanh(b.ks2(torch.cat([diff,knowledge],-1)))
            probabilities.append(b.ks3(pref-difficulty).sigmoid().squeeze(-1))
        expected=(torch.stack(probabilities,1)*self.q).sum(-1)/self.q.sum(-1)
        torch.testing.assert_close(b(self.u,self.e,self.q),expected)

    def test_all_supported_backbones_receive_gate_gradients(self):
        q=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.],[0.,1.,1.],[1.,0.,1.],[1.,1.,1.]])
        for name in ['ncdm','mirt','kancd','kscd']:
            with self.subTest(name=name):
                b=Backbone(name,4,7,3,latent=5)
                net=PaceModel(b,q,'gate',candidates=2)
                pool=torch.tensor([[0,3,5],[0,3,6],[0,1,5],[0,2,6]])
                loss,_=net.objective(self.u,self.e,torch.tensor([1.,0.,1.,0.]),pool)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(sum(float(x.grad.abs().sum()) for x in net.gate.parameters() if x.grad is not None),0)
                self.assertGreater(float(b.student.weight.grad.abs().sum()),0)
                self.assertEqual(sum(x.numel() for x in net.gate.parameters()),193)

    def test_honcd_is_not_silently_substituted(self):
        with self.assertRaises(NotImplementedError):Backbone('honcd',4,7,3)

    def test_committee_ignores_fit_and_tune_labels(self):
        from run import select_committee
        rows=[{'user_id':str(i),'item_id':i,'label':i%2,'score':s}
              for i,s in enumerate([.1,.8,.2,.9,.1,.9])]
        other=[dict(r,score=1-r['score']) for r in rows]
        runs=[{'validation':rows},{'validation':other}]
        manifest={'selection':['0','1','2','3'],'fit':['4'],'tune':['5']}
        selected=select_committee(runs,manifest)
        for r in rows[4:]+other[4:]:r['label']=1-r['label']
        self.assertEqual(selected,select_committee(runs,manifest))
        self.assertEqual(selected,[0])

    def test_inference_never_uses_training_gate_or_attention(self):
        q=torch.ones(7,3)
        model=PaceModel(Backbone('mirt',4,7,3,latent=5),q,'gate',2).eval()
        expected=model(self.u,self.e).detach().clone()
        with torch.no_grad():
            for p in model.gate.parameters():p.fill_(100)
            for p in model.attention.parameters():p.fill_(-100)
        torch.testing.assert_close(model(self.u,self.e),expected)

    def test_legacy_id_conversion_and_pool_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);(r/'data/Eedi').mkdir(parents=True)
            (r/'config.txt').write_text('S,E,K\n4,7,3\n')
            train=[{'user_id':0,'exer_id':0,'knowledge_code':[1],'score':1},
                   {'user_id':1,'exer_id':1,'knowledge_code':[2],'score':0}]
            (r/'data/neg_exer_ours.json').write_text(json.dumps(train))
            for split,item in [('val',3),('test',4)]:
                rows=[{'user_id':1,'logs':[{'exer_id':item,'knowledge_code':[1],'score':1}]},
                      {'user_id':3,'logs':[{'exer_id':item+1,'knowledge_code':[2],'score':0}]}]
                (r/'data/Eedi'/f'{split}_set.json').write_text(json.dumps(rows))
            d=Data(r)
            self.assertEqual(d.rows['validation'][0][0],0)
            self.assertEqual(d.counts['0'],1)
            self.assertEqual(d.counts['2'],0)
            import random
            self.assertNotIn(0,d.pool(0,0,random.Random(42),20))
            self.assertEqual(sum(d.overlap.values()),0)

    def test_end_to_end_freezes_decisions_before_test_export(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repo=root/'data_repo';(repo/'data/Eedi').mkdir(parents=True)
            (repo/'config.txt').write_text('S,E,K\n8,8,3\n')
            train=[{'user_id':u,'exer_id':0,'score':u%2,'knowledge_code':[1]} for u in range(8)]
            (repo/'data/neg_exer_ours.json').write_text(json.dumps(train))
            for split,items in [('val',[3,4]),('test',[5,6])]:
                rows=[{'user_id':u+1,'logs':[{'exer_id':e,'score':j%2,'knowledge_code':[1]}
                                             for j,e in enumerate(items)]} for u in range(8)]
                (repo/'data/Eedi'/f'{split}_set.json').write_text(json.dumps(rows))
            out=root/'output'
            process=subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),
                                    '--repo',str(repo),'--output',str(out),'--backbone','mirt',
                                    '--seeds','2024','--epochs','1','--batch-size','4','--pool-size','4',
                                    '--candidates','2','--evaluate-test'],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['status'],'completed')
            locked=json.loads((out/'committee_locked.json').read_text())
            self.assertFalse(locked['test_used'])
            result=json.loads((out/'test_results.json').read_text())
            self.assertEqual(len(result),4)
            self.assertTrue(all(0<=r['acc_adop']<=1 for r in result))
            self.assertLessEqual((out/'pace_committee_adop_locked.json').stat().st_mtime,
                                 (out/'pace_committee_test_predictions.json').stat().st_mtime)


if __name__=='__main__':unittest.main()

