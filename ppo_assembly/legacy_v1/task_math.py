"""Batched wxyz pose geometry for rewards and physical success checks."""
import torch


def qconj(q):
    return torch.cat((q[..., :1],-q[..., 1:]),dim=-1)


def qmul(a,b):
    aw,av=a[..., :1],a[..., 1:];bw,bv=b[..., :1],b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=-1)),dim=-1)


def apply(q,v):
    t=2*torch.cross(q[..., 1:],v,dim=-1)
    return v+q[..., :1]*t+torch.cross(q[..., 1:],t,dim=-1)


def rotvec(q):
    q=torch.where(q[..., :1]<0,-q,q)
    n=torch.linalg.vector_norm(q[..., 1:],dim=-1,keepdim=True)
    angle=2*torch.atan2(n,q[..., :1].clamp(min=1e-8))
    return q[..., 1:]*torch.where(n>1e-7,angle/n.clamp(min=1e-8),torch.full_like(n,2.))


def axis_quat(vector):
    angle=torch.linalg.vector_norm(vector,dim=-1,keepdim=True)
    scale=torch.where(angle>1e-7,torch.sin(angle/2)/angle.clamp(min=1e-8),torch.full_like(angle,.5))
    return torch.cat((torch.cos(angle/2),vector*scale),dim=-1)


def bolt_metrics(bolt,keel,hole,normal):
    z=torch.zeros_like(normal);z[:,2]=1.
    axis=apply(bolt[:,3:7],z)
    tip=bolt[:,:3]-.028*axis
    d=tip-hole;depth=-(d*normal).sum(-1)
    radial=torch.linalg.vector_norm(d+depth[:,None]*normal,dim=-1)
    cosine=(axis*normal).sum(-1)
    entry=bolt[:,:3]-axis*((bolt[:,:3]-hole)*normal).sum(-1,keepdim=True)/cosine[:,None].clamp(min=1e-6)
    entry_error=torch.linalg.vector_norm(entry-hole,dim=-1)
    axis_error=torch.acos(cosine.clamp(-1,1))
    local=torch.tensor([.0615,.034,-.0205],device=bolt.device,dtype=bolt.dtype).expand_as(normal)
    center=keel[:,:3]+apply(keel[:,3:7],local)
    knormal=apply(keel[:,3:7],z);kc=(axis*knormal).sum(-1)
    deviations=[]
    for side in (-1,1):
        face=center+side*.0015*knormal
        t=((bolt[:,:3]-face)*knormal).sum(-1,keepdim=True)/kc[:,None].clamp(min=1e-6)
        deviations.append(torch.linalg.vector_norm(bolt[:,:3]-axis*t-face,dim=-1))
    keel_radial=torch.maximum(*deviations)
    crosses=(kc>.99)&(((bolt[:,:3]-center)*knormal).sum(-1)>.0015)&(((tip-center)*knormal).sum(-1)<-.0015)
    return dict(depth=depth,radial=radial,entry=entry_error,axis_error=axis_error,
        keel_radial=keel_radial,crosses=crosses,axis=axis)


def centered(m):
    return (m['depth']>.022)&(m['depth']<.0265)&(torch.maximum(m['radial'],m['entry'])<.00035)&(
        m['axis_error']<.015)&m['crosses']&(m['keel_radial']<.00020)


def retained(m):
    return (m['depth']>.020)&(m['depth']<.0265)&(torch.maximum(m['radial'],m['entry'])<.00055)&(
        m['axis_error']<.05)&m['crosses']&(m['keel_radial']<.00035)
