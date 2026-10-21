"""Position/force insertion helpers; all forces here are physical contact forces."""
import torch

TASK_ID='NeroSingleBoltInsertPositionForce-v2'
ACTION_DIM=6
OBSERVATION_DIM=58


def insertion_filter_paths(environment):
    # PhysX matches one body per filter and per environment. A wildcard over
    # all bore sectors silently misses forces or fails the body-count check.
    paths=[environment+'/AssemblyWall/Plate',environment+'/Geometry']
    paths += [environment+f'/AssemblyWall/FrictionBores/Hole4Ring{ring}Sector{sector:02d}'
        for ring in (1,2) for sector in range(32)]
    return paths


def axial_velocity(requested,force,reference,contact,maximum,gain,side_yield):
    """Force-limited admittance in the insertion axis; allow active withdrawal.

    force[:,0] is positive resistance along the outward wall normal. The
    lateral components are measured in the hole plane. In free space there
    is no force chasing. Above the reference force, the hand yields outward.
    """
    resistance=force[:,0].clamp(min=0.)
    lateral=torch.linalg.vector_norm(force[:,1:],dim=-1)
    response=gain*(reference-resistance)
    response=torch.minimum(response,-gain*(lateral-side_yield).clamp(min=0.)+maximum)
    limited=torch.minimum(requested,response)
    return torch.where(contact&(requested>=0.),limited,requested).clamp(-maximum,maximum)


def force_costs(force,previous,reference,contact,tracking,axial_soft,scale,side_scale):
    """Named, nonnegative costs; never award merely pushing on the wall."""
    axial=force[:,0].clamp(min=0.)
    return dict(
        tracking=contact.float()*tracking.float()*((axial-reference)/scale).square(),
        lateral=force[:,1:].square().sum(-1)/(side_scale**2),
        overload=((axial-axial_soft).clamp(min=0.)/scale).square(),
        variation=(force-previous).square().sum(-1)/(scale**2))
