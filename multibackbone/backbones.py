"""Response functions and explicit augmentation adapters.

MIRT: EduCDM/MIRT/MIRT.py; KaNCD (gmf): EduCDM/KaNCD/KaNCD.py.
KSCD interaction: ECNU-ILOG/InsCD, inscd/interfunc/kscd.py.
These are transparent reimplementations, not untouched upstream packages.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PositiveLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.abs(), self.bias)


class Backbone(nn.Module):
    def __init__(self, name, students, items, concepts, latent=32):
        super().__init__()
        if name == 'honcd':
            raise NotImplementedError('HO-NCD is blocked pending upstream forward/group/response-time audit; no surrogate is substituted.')
        if name not in {'ncdm', 'mirt', 'kancd', 'kscd'}:
            raise ValueError(name)
        self.name, self.concepts = name, concepts
        self.dim = concepts if name == 'ncdm' else latent
        self.student = nn.Embedding(students, self.dim)
        self.item = nn.Embedding(items, self.dim)
        self.discrimination = nn.Embedding(items, 1)
        if name == 'mirt':
            # item represents raw a; discrimination stores the scalar b.
            return
        if name in {'kancd', 'kscd'}:
            self.knowledge = nn.Parameter(torch.empty(concepts, self.dim))
            nn.init.xavier_normal_(self.knowledge)
        if name == 'kancd':
            self.ability_map = nn.Linear(self.dim, 1)
            self.difficulty_map = nn.Linear(self.dim, 1)
        if name in {'ncdm', 'kancd'}:
            hidden = (512, 256) if name == 'ncdm' else (256, 128)
            self.head = nn.ModuleList([PositiveLinear(concepts, hidden[0]),
                                       PositiveLinear(hidden[0], hidden[1]), PositiveLinear(hidden[1], 1)])
            self.dropout = nn.Dropout(.5)
        else:
            self.ks1 = nn.Linear(concepts+self.dim, concepts, bias=False)
            self.ks2 = nn.Linear(concepts+self.dim, concepts, bias=False)
            self.ks3 = nn.Linear(concepts, 1)
        for key, parameter in self.named_parameters():
            if parameter.ndim >= 2 and key != 'knowledge':
                nn.init.xavier_normal_(parameter)

    def item_features(self, item):
        x = self.item(item)
        return x.sigmoid() if self.name == 'ncdm' else x

    def concept_difficulty(self, features, item):
        if self.name == 'mirt':
            return self.discrimination(item)  # scalar difficulty, not discrimination a
        if self.name == 'ncdm':
            return features
        if self.name == 'kancd':
            return self.difficulty_map(features.unsqueeze(-2)*self.knowledge).squeeze(-1).sigmoid()
        return (features @ self.knowledge.T).sigmoid()

    def similarity_features(self, item):
        f = self.item_features(item)
        # MIRT has no concept difficulty vector: use positive discrimination a.
        return F.softplus(f) if self.name == 'mirt' else self.concept_difficulty(f, item)

    def sampling_discrimination(self, item):
        return F.softplus(self.item(item)) if self.name == 'mirt' else self.discrimination(item).sigmoid()

    def score_with_features(self, user, item, q, features):
        ability = self.student(user)
        if self.name == 'mirt':
            return ((ability*F.softplus(features)).sum(-1)-self.discrimination(item).squeeze(-1)).sigmoid()
        if self.name == 'kscd':
            stu = (ability @ self.knowledge.T).sigmoid()
            diff = (features @ self.knowledge.T).sigmoid()
            k = self.knowledge.unsqueeze(0).expand(len(user), -1, -1)
            s = stu.unsqueeze(1).expand(-1, self.concepts, -1)
            d = diff.unsqueeze(1).expand_as(s)
            out = self.ks3(torch.tanh(self.ks1(torch.cat([s,k],-1)))-
                           torch.tanh(self.ks2(torch.cat([d,k],-1)))).sigmoid().squeeze(-1)
            if (q.sum(-1) == 0).any():
                raise ValueError('KSCD requires a nonempty Q row')
            return (out*q).sum(-1)/q.sum(-1)
        if self.name == 'ncdm':
            ability = ability.sigmoid()
        else:
            ability = self.ability_map(ability.unsqueeze(1)*self.knowledge).squeeze(-1).sigmoid()
        diff = self.concept_difficulty(features, item)
        x = self.discrimination(item).sigmoid()*(ability-diff)*q
        activation = torch.sigmoid if self.name == 'ncdm' else torch.tanh
        for layer in self.head[:2]:
            x = self.dropout(activation(layer(x)))
        return self.head[2](x).sigmoid().squeeze(-1)

    def forward(self, user, item, q):
        return self.score_with_features(user, item, q, self.item_features(item))

    @torch.no_grad()
    def clip(self):
        if self.name == 'kscd':
            for layer in [self.ks1, self.ks2, self.ks3]:
                layer.weight.clamp_(min=0)


class PaceModel(nn.Module):
    def __init__(self, backbone, qmatrix, variant='original', candidates=20):
        super().__init__()
        if variant not in {'original', 'fixed', 'gate'}:
            raise ValueError(variant)
        self.backbone, self.variant, self.candidates = backbone, variant, candidates
        self.register_buffer('qmatrix', qmatrix)
        if variant != 'original':
            heads = 2 if backbone.dim % 2 == 0 else 1
            self.attention = nn.MultiheadAttention(backbone.dim, heads, batch_first=True)
        if variant == 'gate':
            self.gate = nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self, user, item):
        return self.backbone(user, item, self.qmatrix[item])

    def objective(self, user, item, labels, pool=None):
        q = self.qmatrix[item]
        positive = self.backbone(user,item,q)
        observed = F.binary_cross_entropy(positive,labels)
        if self.variant == 'original':
            return observed, {'observed':float(observed.detach())}
        if pool is None or pool.ndim != 2 or not pool.shape[1]:
            raise ValueError('Augmentation requires an admissible unobserved candidate pool')
        anchor = self.backbone.item_features(item)
        features = self.backbone.item_features(pool)
        discr = self.backbone.sampling_discrimination(pool)
        ad = self.backbone.sampling_discrimination(item)
        with torch.no_grad():
            p1 = torch.softmax((anchor.unsqueeze(1)*features).sum(-1),dim=1)
            p2 = torch.softmax((ad.unsqueeze(1)*discr).sum(-1),dim=1)
            draws = torch.multinomial((p1+p2)/2,self.candidates,replacement=True)
        candidate = pool.gather(1,draws)
        f = self.backbone.item_features(candidate)
        a_sim = self.backbone.similarity_features(item)
        c_sim = self.backbone.similarity_features(candidate)
        rho = .5*F.cosine_similarity(a_sim.unsqueeze(1),c_sim,dim=-1)+.5*F.cosine_similarity(q.unsqueeze(1),self.qmatrix[candidate],dim=-1)
        response = torch.sigmoid(8*(rho-.5))
        response = torch.where(labels.unsqueeze(1)==1,response,1-response)
        flat_user=user.unsqueeze(1).expand_as(candidate).reshape(-1)
        flat_q=q.unsqueeze(1).expand(-1,self.candidates,-1).reshape(-1,q.shape[1])
        neg = self.backbone.score_with_features(flat_user,candidate.reshape(-1),flat_q,f.reshape(-1,f.shape[-1])).reshape_as(response)
        if self.variant=='gate':
            inputs=torch.stack([rho,positive.unsqueeze(1).expand_as(rho),neg,(rho-.5).abs()],-1)
            conf=self.gate(inputs).squeeze(-1).sigmoid()
        else:
            conf=2*(response-.5).abs()
        weights=conf/(conf.sum(0,keepdim=True)+1e-8)
        pseudo=(weights*(neg-response).square()).sum(0).mean()
        sequence=torch.cat([anchor.unsqueeze(1),f],1)
        fused,_=self.attention(sequence,sequence,sequence,need_weights=False)
        seqitems=torch.cat([item.unsqueeze(1),candidate],1)
        seq_user=user.unsqueeze(1).expand_as(seqitems).reshape(-1)
        seq_q=q.unsqueeze(1).expand(-1,seqitems.shape[1],-1).reshape(-1,q.shape[1])
        fused_score=self.backbone.score_with_features(seq_user,seqitems.reshape(-1),seq_q,fused.reshape(-1,fused.shape[-1])).reshape_as(seqitems)
        change=torch.softmax((positive.unsqueeze(1)-fused_score).abs(),dim=1)
        # Keep legacy batch-mean directional ranking behavior, shared across variants.
        margins=positive.unsqueeze(1)-fused_score
        right=-F.logsigmoid(margins).mean(0)
        wrong=-F.logsigmoid(-margins).mean(0)
        bpr=(change*torch.where(labels.unsqueeze(1)==1,right,wrong)).mean(0).sum()
        da=self.backbone.concept_difficulty(anchor,item)
        dc=self.backbone.concept_difficulty(f,candidate)
        difficulty=F.relu((dc-da.unsqueeze(1))*(neg-positive.unsqueeze(1)).unsqueeze(-1)).mean(dim=(0,2)).sum()
        loss=observed+pseudo+.1*bpr+difficulty
        return loss,{k:float(v.detach()) for k,v in [('observed',observed),('pseudo',pseudo),('bpr',bpr),('difficulty',difficulty)]}

