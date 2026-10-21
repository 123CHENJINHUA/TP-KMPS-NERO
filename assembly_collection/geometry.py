"""Pose math and measured assembly geometry; no Isaac Sim dependency."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FINGER_CENTER = np.array([0., 0., .125])
KEEL_GRIP = np.array([-.040, 0., 0.])
HOLES = np.array([[x, y, -.0205] for x in (-.0615, .0615) for y in (-.034, .034)])
HOLE_RADII = (.006, .006, .00525, .00525)


def screw_order(first=4, count=4):
    """One-based order: right top/bottom first, then the left column."""
    if first not in (3,4) or not 1<=count<=4:
        raise ValueError('Right-column handoff requires first screw 3 or 4 and count 1..4')
    return ([first]+[i for i in (4,3,1,2) if i!=first])[:count]


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]])


def pose(position, rot):
    return np.r_[position, rot.as_quat()[[3, 0, 1, 2]]]


def compose(a, b):
    return pose(np.asarray(a[:3]) + rotation(a).apply(b[:3]), rotation(a) * rotation(b))


def inverse(a):
    r = rotation(a).inv()
    return pose(r.apply(-np.asarray(a[:3])), r)


def canonical_screw_target(target, grip_relative):
    """Choose the round bolt's free axial roll to preserve the jaw direction.

    In the nominal wall bolt frame, +X is the desired gripper closing axis.
    Arbitrary bolt spin acquired in the rack must not clock the wide hand.
    """
    closing=rotation(grip_relative).apply([0.,1.,0.])
    angle=-np.arctan2(closing[1],closing[0])
    return pose(target[:3],rotation(target)*Rotation.from_euler('z',angle))


def axial_alignment_target(wanted, actual):
    """Align a round bolt's axis while preserving its unconstrained roll."""
    source=rotation(actual).apply([0.,0.,1.]);target=rotation(wanted).apply([0.,0.,1.])
    axis=np.cross(source,target);sine=np.linalg.norm(axis);cosine=np.clip(np.dot(source,target),-1.,1.)
    if sine>1e-9:
        correction=Rotation.from_rotvec(axis/sine*np.arctan2(sine,cosine))
    elif cosine>0:
        correction=Rotation.identity()
    else:
        basis=np.array([1.,0.,0.]) if abs(source[0])<.9 else np.array([0.,1.,0.])
        axis=np.cross(source,basis);axis/=np.linalg.norm(axis)
        correction=Rotation.from_rotvec(np.pi*axis)
    return pose(wanted[:3],correction*rotation(actual))


def interpolate(a, b, fraction):
    s = np.clip(fraction, 0., 1.)
    s = s * s * (3 - 2 * s)
    r = Slerp([0, 1], Rotation.from_quat([np.asarray(a)[[4,5,6,3]], np.asarray(b)[[4,5,6,3]]]))([s])[0]
    return pose((1-s) * np.asarray(a[:3]) + s * np.asarray(b[:3]), r)


def errors(actual, wanted):
    return float(np.linalg.norm(actual[:3]-wanted[:3])), float((rotation(actual)*rotation(wanted).inv()).magnitude())


def top_grasp(obj, local_point=(0, 0, 0), yaw=0., tilt=0.):
    r = Rotation.from_euler('y', tilt) * Rotation.from_euler('z', yaw) * Rotation.from_euler('x', np.pi)
    return pose(np.asarray(obj[:3]) + rotation(obj).apply(local_point) - r.apply(FINGER_CENTER), r)


def keel_grasp(obj, local_x=-.040, tilt=0.):
    """A rail grasp expressed in the keel frame, including its starting yaw."""
    r=Rotation.from_euler('y',tilt)*Rotation.from_euler('x',np.pi)
    local=pose(np.array([local_x,0.,0.])-r.apply(FINGER_CENTER),r)
    return compose(obj,local)


def wall_targets(wall):
    # The panel's local X is horizontal and Z is up; its front normal is -Y.
    wr = rotation(wall)
    normal = wr.apply([0, -1, 0])
    horizontal = wr.apply([1, 0, 0])
    up = wr.apply([0, 0, 1])
    kr = Rotation.from_matrix(np.column_stack((horizontal, up, normal)))
    front_center = np.asarray(wall[:3]) + wr.apply([0, -.015, .1645])
    # 0.4 mm intentional clearance from wall while the holding arm aligns.
    keel = pose(front_center + normal * (.022 + .0004), kr)
    holes = np.array([front_center + horizontal*x + up*y for x,y,_ in HOLES])
    # Bolt +Z points out of the wall; its negative-Z shaft inserts inward.
    # Clock the bolt so the second gripper closes vertically at the wall.
    sr = Rotation.from_matrix(np.column_stack((-up, horizontal, normal)))
    # Keep the head 0.3 mm clear of the plate face. Otherwise the head/plate
    # contact competes with the finger contact at the exact seating plane.
    screws = np.array([pose(h + normal * (.003 + .0004 + .0003), sr) for h in holes])
    return keel, holes, screws, normal


def insertion_metrics(screw, hole, normal):
    axis=rotation(screw).apply([0,0,1])
    tip = screw[:3] - axis*.028
    delta = tip-hole
    depth = -float(np.dot(delta, normal))
    radial = float(np.linalg.norm(delta + depth*normal))
    cosine=float(np.dot(axis,normal))
    axis_error = float(np.arccos(np.clip(cosine, -1, 1)))
    entry=screw[:3]-axis*(np.dot(screw[:3]-hole,normal)/max(cosine,1e-8))
    entry_error=float(np.linalg.norm(entry-hole))
    return dict(depth_m=depth, radial_error_m=radial, entry_radial_error_m=entry_error,axis_error_rad=axis_error)


def keel_hole_metrics(screw, keel, index, bearing_radius=.004):
    """Check that the shaft crosses both faces of the actual moving keel hole."""
    kr=rotation(keel)
    center=keel[:3]+kr.apply(HOLES[index])
    normal=kr.apply([0,0,1])
    axis=rotation(screw).apply([0,0,1])
    cosine=float(np.dot(axis,normal))
    radial=[]
    for side in (-1,1):
        face=center+side*.0015*normal
        entry=screw[:3]-axis*(np.dot(screw[:3]-face,normal)/max(cosine,1e-8))
        radial.append(float(np.linalg.norm(entry-face)))
    tip=screw[:3]-.028*axis
    return dict(keel_hole_radial_error_m=max(radial),keel_hole_clearance_m=HOLE_RADII[index]-bearing_radius,
        keel_shaft_crosses_plate=bool(cosine>.99 and np.dot(screw[:3]-center,normal)>.0015
            and np.dot(tip-center,normal)<-.0015))
