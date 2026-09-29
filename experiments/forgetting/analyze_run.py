#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--lead",type=int,default=1); a=ap.parse_args()
    root=Path(a.run); out=root/"analysis"; out.mkdir(exist_ok=True); s=pd.read_csv(root/"summary.csv")
    base_ck=int(s.query("task == 0").checkpoint.max())
    records=[]
    base={}
    for ck in s.checkpoint:
        for layer in ("hidden1","hidden2","logits","gradient_last_layer"):
            z=np.load(root/"checkpoints"/f"ckpt_{ck:04d}"/f"layer_{layer}.npz",allow_pickle=False)
            if ck==base_ck:base[layer]=z["magnitude"].copy()
            if layer in base:
                m=z["magnitude"]; b=base[layer]
                records.append({"checkpoint":ck,"layer":layer,"l1_profile_drift":float(np.mean(np.abs(m-b))),"l2_profile_drift":float(np.sqrt(np.mean((m-b)**2))),"max_profile_drift":float(np.max(np.abs(m-b)))})
    drift=pd.DataFrame(records); drift.to_csv(out/"profile_drift.csv",index=False)
    wide=drift.pivot(index="checkpoint",columns="layer",values="l1_profile_drift").add_prefix("mag_drift_").reset_index()
    q=s.merge(wide,on="checkpoint",how="left"); target=1-q["accuracy_task0"].shift(-a.lead)
    features=[x for x in q if x.startswith("mag_drift_") or x.endswith("_displacement") or x.endswith("_magnitude_mid")]
    corr=[]
    for f in features:corr.append({"feature":f,"lead_checkpoints":a.lead,"pearson_with_future_task0_forgetting":q[f].corr(target),"n":int((q[f].notna()&target.notna()).sum())})
    pd.DataFrame(corr).to_csv(out/"lead_lag_correlations.csv",index=False)
    fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for t in range(3):ax[0].plot(s.global_epoch,s[f"accuracy_task{t}"],marker='o',label=f"task {t}")
    ax[0].set_ylabel("reference accuracy"); ax[0].legend(); ax[0].grid(alpha=.25)
    for col in wide.columns[1:]:ax[1].plot(q.global_epoch,q[col],marker='o',label=col.replace('mag_drift_',''))
    ax[1].set_ylabel("L1 magnitude-profile drift"); ax[1].set_xlabel("global epoch"); ax[1].legend(); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"overview.png",dpi=160); plt.close(fig)
    manifest_path=root/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    import hashlib
    for p in sorted(out.glob("*")):
        if p.is_file():manifest["sha256"][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"analysis":str(out),"baseline_checkpoint":base_ck,"rows":len(s)},indent=2))
if __name__=="__main__":main()
