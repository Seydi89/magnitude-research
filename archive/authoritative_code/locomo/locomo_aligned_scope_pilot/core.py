import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
TAUS=np.array([.25,.5,1.,2.,4.,8.]); RIDGE=1e-8

def magnitude(k):
 return float(np.ones(len(k))@cho_solve(cho_factor(k,lower=True),np.ones(len(k)))) if len(k) else 0.

def kernels(x,unit):
 return np.exp(-TAUS[:,None,None]*cdist(x,x)/unit)+RIDGE*np.eye(len(x))[None]

def gains(k,s):
 n=len(k)
 if not s:return 1/np.diag(k)
 cross=k[np.ix_(s,range(n))];v=cho_solve(cho_factor(k[np.ix_(s,s)],lower=True),cross)
 den=np.diag(k)-np.sum(cross*v,axis=0);remain=list(set(range(n))-set(s))
 if np.any(den[remain]<=0):raise ArithmeticError('Invalid Schur complement')
 out=np.zeros(n);out[remain]=(1-v[:,remain].sum(0))**2/den[remain];return out

def scope(x,window=6,decay=.8):
 if len(x)<window+1:raise ValueError('Insufficient history')
 spread=[];centers=[]
 for end in range(window,len(x)+1):
  z=x[end-window:end];mu=z.mean(0);centers.append(mu)
  spread.append(float(np.mean(np.sum((z-mu)**2,axis=1))))
 change=np.diff(spread);beta=decay**np.arange(len(change)-1,-1,-1);beta/=beta.sum()
 return dict(trend=float(beta@change),unweighted=float(change.mean()),level=spread[-1],
             centroid_shift=float(np.linalg.norm(centers[-1]-centers[0])),
             spread=spread,beta=beta.tolist())

def scale_weights(u,sign=1,strength=1):
 z=sign*strength*np.clip(u,-3,3)*np.linspace(-1,1,len(TAUS));z-=z.max();w=np.exp(z);return w/w.sum()

def select(k,cost,rel,budget,w,lam=.5,baseline=None):
 s=[];spent=0;trace=[];n=len(cost);norm=np.array([magnitude(z) for z in k])
 r=(rel-rel.min())/max(float(np.ptp(rel)),1e-12)
 while True:
  fit=[j for j in range(n) if j not in s and spent+cost[j]<=budget]
  if not fit:break
  if not s or baseline=='relevance':scores=r
  elif baseline=='mmr':scores=.5*r-.5*k[2][:,s].max(1)
  else:scores=lam*r+(1-lam)*(w@(np.array([gains(z,s) for z in k])*n/norm[:,None]))
  j=max(fit,key=lambda i:(scores[i],rel[i],-i));s.append(j);spent+=int(cost[j])
  trace.append(dict(id_index=j,score=float(scores[j]),tokens=int(cost[j])))
 return s,spent,trace
