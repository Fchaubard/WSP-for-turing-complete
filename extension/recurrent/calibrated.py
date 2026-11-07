"""Passive gradient control with a fixed binary64 hidden-state scale."""
import json
from pathlib import Path
from run import ROOT, evaluate, fit, setup, torch, nn


class ScaledSum(nn.Module):
    task = 'sum'
    def __init__(self, seed,scale):
        super().__init__()
        self.scale = scale
        self.a = nn.Parameter(torch.empty(1,1))
        self.b = nn.Parameter(torch.empty(1,1))
        gen = torch.Generator().manual_seed(seed)
        nn.init.xavier_uniform_(self.a,generator=gen)
        nn.init.xavier_uniform_(self.b,generator=gen)

    def forward(self,x,T):
        h = torch.zeros_like(x[:,0,0])
        result = []
        for i in range(T):
            h = torch.tanh(self.a[0,0]*h + (self.scale*self.b[0,0])*x[:,i,0])
            result.append(h/self.scale)
        return torch.stack(result,dim=1)


def main():
    setup()
    rows = []
    for scale in [1.,2.**-40]:
        for seed in [101,202,303,404,505]:
            model = ScaledSum(seed,scale)
            initial = [float(p.item()) for p in model.parameters()]
            record = fit(model,seed,5000,200)
            final = [float(p.item()) for p in model.parameters()]
            scores = evaluate(model)
            rows.append(dict(scale=scale,seed=seed,initial=initial,final=final,
                             fit=record,metrics=scores))
            print(scale,seed,final,scores['joint']['exact_all'],scores['joint']['nrmse'],flush=True)
    (ROOT/'calibrated.json').write_text(json.dumps(rows,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
