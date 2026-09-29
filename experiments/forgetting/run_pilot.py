#!/usr/bin/env python3
"""Instrumented continual-learning pilot for multiscale metric-space magnitude."""
from __future__ import annotations

import argparse, hashlib, json, platform, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Config:
    output: str = "runs/pilot"
    seed: int = 17
    samples_per_class: int = 240
    reference_per_class: int = 24
    hidden1: int = 32
    hidden2: int = 16
    tasks: int = 3
    classes_per_task: int = 2
    epochs_per_task: int = 35
    checkpoint_every: int = 5
    batch_size: int = 64
    learning_rate: float = 0.06
    weight_decay: float = 1e-4
    scales: int = 33
    regularization: float = 1e-7
    store_kernels: bool = False


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, dims, rng):
        self.p = {}
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.p[f"W{i}"] = rng.normal(0, np.sqrt(2/a), (a, b))
            self.p[f"b{i}"] = np.zeros(b)

    def forward(self, x):
        h1 = np.tanh(x @ self.p["W1"] + self.p["b1"])
        h2 = np.tanh(h1 @ self.p["W2"] + self.p["b2"])
        logits = h2 @ self.p["W3"] + self.p["b3"]
        return logits, {"hidden1": h1, "hidden2": h2, "logits": logits}

    def step(self, x, y, lr, wd):
        logits, a = self.forward(x); p = softmax(logits); n = len(x)
        dz = p; dz[np.arange(n), y] -= 1; dz /= n
        grads = {}
        grads["W3"] = a["hidden2"].T @ dz + wd*self.p["W3"]; grads["b3"] = dz.sum(0)
        dh2 = (dz @ self.p["W3"].T) * (1-a["hidden2"]**2)
        grads["W2"] = a["hidden1"].T @ dh2 + wd*self.p["W2"]; grads["b2"] = dh2.sum(0)
        dh1 = (dh2 @ self.p["W2"].T) * (1-a["hidden1"]**2)
        grads["W1"] = x.T @ dh1 + wd*self.p["W1"]; grads["b1"] = dh1.sum(0)
        for k in self.p: self.p[k] -= lr*grads[k]


def make_data(c, rng):
    k = c.tasks*c.classes_per_task
    angles = np.linspace(0, 2*np.pi, k, endpoint=False)
    centers = np.c_[3.2*np.cos(angles), 3.2*np.sin(angles)]
    xs=[]; ys=[]
    for label, mu in enumerate(centers):
        x = rng.normal(mu, [1.0, .65], (c.samples_per_class, 2))
        # Deterministic nonlinear lift makes the benchmark nontrivial but inspectable.
        x = np.c_[x, x[:,0]*x[:,1]/4, np.sin(x[:,0]), np.cos(x[:,1])]
        xs.append(x); ys.append(np.full(len(x), label))
    X=np.vstack(xs); y=np.concatenate(ys).astype(int)
    perm=rng.permutation(len(X)); return X[perm],y[perm],centers


def pdist(x, cosine=False):
    if cosine:
        n=np.linalg.norm(x,axis=1,keepdims=True); z=x/np.maximum(n,1e-12)
        return np.clip(1-z@z.T,0,2)
    q=(x*x).sum(1); return np.sqrt(np.maximum(q[:,None]+q[None,:]-2*x@x.T,0))


def effective_rank(x):
    s=np.linalg.svd(x-x.mean(0),compute_uv=False)**2
    if s.sum()==0:return 0.,0.
    p=s/s.sum(); er=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))
    return er,float(s.sum()**2/np.maximum((s*s).sum(),1e-15))


def magnitude_bundle(x, scales_n, reg, include_kernels, labels=None, classifier=None):
    d_euc=pdist(x); d_cos=pdist(x,True)
    nz=d_euc[d_euc>1e-12]; med=float(np.median(nz)) if len(nz) else 1.
    scales=np.concatenate(([0.0],np.logspace(-2,2,scales_n-1)/med))
    mags=[]; weights=[]; weights_reg=[]; weights_nn=[]; cond=[]; residual=[]; kernels=[]
    one=np.ones(len(x)); eye=np.eye(len(x))
    for t in scales:
        Z=np.exp(-t*d_euc)
        A=Z+reg*eye
        try: w=np.linalg.solve(A,one)
        except np.linalg.LinAlgError: w=np.linalg.lstsq(A,one,rcond=1e-10)[0]
        wr=np.linalg.solve(Z+max(reg,1e-5)*eye,one)
        wn=np.maximum(wr,0); wn=wn/wn.sum() if wn.sum() else np.ones(len(x))/len(x)
        mags.append(w.sum()); weights.append(w); weights_reg.append(wr); weights_nn.append(wn)
        cond.append(np.linalg.cond(A)); residual.append(np.linalg.norm(A@w-one))
        if include_kernels:kernels.append(Z.astype(np.float32))
    mags=np.asarray(mags); logscale=np.log(np.maximum(scales,1e-12))
    slope=np.gradient(mags,logscale)
    er,pr=effective_rank(x); tri=d_euc[np.triu_indices(len(x),1)]
    extra={}
    if labels is not None:
        same=labels[:,None]==labels[None,:]; off=~np.eye(len(labels),dtype=bool)
        wi=d_euc[same&off]; be=d_euc[(~same)&off]
        extra.update(within_class_distance=float(wi.mean()) if len(wi) else 0.,
                     between_class_distance=float(be.mean()) if len(be) else 0.,
                     separation_ratio=float(be.mean()/max(wi.mean(),1e-12)) if len(wi) and len(be) else 0.)
        if classifier is not None and classifier.shape[0]==x.shape[1]:
            aligns=[]
            for lab in np.unique(labels):
                cen=x[labels==lab].mean(0); w=classifier[:,int(lab)]
                aligns.append(float(cen@w/max(np.linalg.norm(cen)*np.linalg.norm(w),1e-12)))
            extra["mean_classifier_alignment"]=float(np.mean(aligns))
    out=dict(representations=x.astype(np.float32),distance_euclidean=d_euc.astype(np.float32),
             distance_cosine=d_cos.astype(np.float32),scales=scales,magnitude=mags,
             magnitude_weights=np.asarray(weights),regularized_weights=np.asarray(weights_reg),
             nonnegative_weights=np.asarray(weights_nn),condition_number=np.asarray(cond),
             solve_residual=np.asarray(residual),profile_slope=slope,
             kernels=np.asarray(kernels) if include_kernels else np.empty((0,)),
             scalar_json=np.array(json.dumps({"median_distance":med,"effective_rank":er,
                 "participation_ratio":pr,"pairwise_mean":float(tri.mean()) if len(tri) else 0,
                 "pairwise_std":float(tri.std()) if len(tri) else 0,
                 "profile_auc_log_scale":float(np.trapezoid(mags[1:],logscale[1:])),
                 "max_slope_scale":float(scales[int(np.argmax(slope))]),**extra})))
    return out


def metrics(model,X,y,seen):
    logits,_=model.forward(X); p=softmax(logits); pred=p.argmax(1)
    loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12))
    ans={"accuracy_all":float((pred==y).mean()),"loss_all":float(loss.mean()),
         "prediction_entropy_all":float(np.mean(-np.sum(p*np.log(np.maximum(p,1e-12)),axis=1)))}
    for t in range(3):
        m=np.isin(y,[2*t,2*t+1]); ans[f"accuracy_task{t}"]=float((pred[m]==y[m]).mean()); ans[f"loss_task{t}"]=float(loss[m].mean())
    old=np.isin(y,seen); ans["accuracy_seen"]=float((pred[old]==y[old]).mean()); ans["loss_seen"]=float(loss[old].mean())
    return ans,logits,p,pred,loss


def save_checkpoint(root,idx,task,epoch,model,Xr,yr,ids,c,baseline_repr):
    d=root/"checkpoints"/f"ckpt_{idx:04d}"; (d/"groups").mkdir(parents=True,exist_ok=True)
    seen=list(range((task+1)*c.classes_per_task))
    met,logits,p,pred,loss=metrics(model,Xr,yr,seen)
    _,layers=model.forward(Xr)
    delta=p.copy(); delta[np.arange(len(yr)),yr]-=1
    # Per-example final-layer gradients: vec(h2 outer dlogits) concatenated with bias gradient.
    gw=np.einsum('bi,bj->bij',layers["hidden2"],delta).reshape(len(yr),-1)
    layers["gradient_last_layer"]=np.c_[gw,delta]
    np.savez_compressed(d/"model.npz",**model.p)
    np.savez_compressed(d/"predictions.npz",sample_id=ids,labels=yr,logits=logits,probabilities=p,predictions=pred,loss=loss)
    rows={"checkpoint":idx,"task":task,"epoch_in_task":epoch,"global_epoch":task*c.epochs_per_task+epoch,**met}
    for lname,h in layers.items():
        classifier=model.p["W3"] if lname=="hidden2" else None
        bundle=magnitude_bundle(h,c.scales,c.regularization,c.store_kernels,yr,classifier)
        displacement=float(np.mean(np.linalg.norm(h-baseline_repr[lname],axis=1))) if lname in baseline_repr else 0.
        meta={"checkpoint":idx,"task":task,"epoch":epoch,"layer":lname,"group":"all","displacement_from_post_task0":displacement}
        np.savez_compressed(d/f"layer_{lname}.npz",sample_id=ids,labels=yr,metadata_json=np.array(json.dumps(meta)),**bundle)
        rows[f"{lname}_magnitude_mid"]=float(bundle["magnitude"][len(bundle["magnitude"])//2]); rows[f"{lname}_displacement"]=displacement
        for group_kind, values in (("task",range(c.tasks)),("class",range(c.tasks*c.classes_per_task))):
            for g in values:
                mask=(yr//c.classes_per_task==g) if group_kind=="task" else (yr==g)
                if mask.sum()<2:continue
                gb=magnitude_bundle(h[mask],c.scales,c.regularization,c.store_kernels,yr[mask],classifier)
                gm={"checkpoint":idx,"task":task,"epoch":epoch,"layer":lname,"group_kind":group_kind,"group":int(g)}
                np.savez_compressed(d/"groups"/f"{lname}_{group_kind}_{g}.npz",sample_id=ids[mask],labels=yr[mask],metadata_json=np.array(json.dumps(gm)),**gb)
    (d/"metrics.json").write_text(json.dumps(rows,indent=2))
    return rows,layers


def hashes(root):
    out={}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name!="manifest.json":out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main():
    ap=argparse.ArgumentParser();
    for k,v in asdict(Config()).items():
        flag="--"+k.replace("_","-")
        if isinstance(v,bool):ap.add_argument(flag,action="store_true",default=v)
        else:ap.add_argument(flag,type=type(v),default=v)
    a=ap.parse_args(); c=Config(**vars(a)); root=Path(c.output); (root/"checkpoints").mkdir(parents=True,exist_ok=True)
    (root/"config.json").write_text(json.dumps(asdict(c),indent=2)); rng=np.random.default_rng(c.seed)
    X,y,centers=make_data(c,rng)
    ref_idx=np.concatenate([np.where(y==k)[0][:c.reference_per_class] for k in range(c.tasks*c.classes_per_task)])
    train_mask=np.ones(len(y),bool); train_mask[ref_idx]=False; Xr,yr=X[ref_idx],y[ref_idx]; ids=ref_idx
    np.savez_compressed(root/"dataset.npz",X=X,y=y,reference_indices=ref_idx,train_mask=train_mask,class_centers=centers)
    model=MLP([X.shape[1],c.hidden1,c.hidden2,c.tasks*c.classes_per_task],rng)
    summary=[]; ck=0; baseline={}
    row,layers=save_checkpoint(root,ck,0,0,model,Xr,yr,ids,c,baseline); summary.append(row); ck+=1
    for task in range(c.tasks):
        m=train_mask & np.isin(y,[2*task,2*task+1]); xt,yt=X[m],y[m]
        for epoch in range(1,c.epochs_per_task+1):
            order=rng.permutation(len(xt))
            for j in range(0,len(order),c.batch_size):model.step(xt[order[j:j+c.batch_size]],yt[order[j:j+c.batch_size]],c.learning_rate,c.weight_decay)
            if epoch%c.checkpoint_every==0 or epoch==c.epochs_per_task:
                row,layers=save_checkpoint(root,ck,task,epoch,model,Xr,yr,ids,c,baseline); summary.append(row)
                if task==0 and epoch==c.epochs_per_task:baseline={k:v.copy() for k,v in layers.items()}
                ck+=1
    pd.DataFrame(summary).to_csv(root/"summary.csv",index=False)
    pd.DataFrame([{"checkpoint":r["checkpoint"],"task":r["task"],"epoch":r["epoch_in_task"],"path":f"ckpt_{r['checkpoint']:04d}"} for r in summary]).to_csv(root/"checkpoints"/"index.csv",index=False)
    manifest={"format_version":"1.0","created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"python":platform.python_version(),"numpy":np.__version__,"config":asdict(c)}
    manifest["sha256"]=hashes(root); (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Completed {len(summary)} checkpoints in {root}")

if __name__=="__main__": main()
