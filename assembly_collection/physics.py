"""Assembly-specific contact geometry, authored only on the live trial stage."""
import math
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade


def preserve_keel_holes(stage, path):
    """Compound convex plate segments and channel rails preserve four openings.

    The supplied visual mesh stays unchanged. The collision approximation is
    explicit: 3 mm plate and three rectangular segments per channel rail.
    """
    body = stage.GetPrimAtPath(path)
    UsdPhysics.CollisionAPI(body).CreateCollisionEnabledAttr(False)
    base = path + '/AssemblyCollision'
    container = UsdGeom.Xform.Define(stage, base)
    UsdGeom.Imageable(container.GetPrim()).CreateVisibilityAttr('invisible')
    count = 0

    def box(x0, x1, y0, y1, z0, z1):
        nonlocal count
        count += 1
        c = UsdGeom.Cube.Define(stage, base + f'/Block{count:03d}')
        c.CreateSizeAttr(1.)
        t = UsdGeom.Xformable(c.GetPrim())
        t.AddTranslateOp().Set(Gf.Vec3d((x0+x1)/2, (y0+y1)/2, (z0+z1)/2))
        t.AddScaleOp().Set(Gf.Vec3f(x1-x0, y1-y0, z1-z0))
        UsdPhysics.CollisionAPI.Apply(c.GetPrim()).CreateCollisionEnabledAttr(True)

    xs = [-.075, -.0715, -.0515, .0515, .0715, .075]
    ys = [-.049, -.044, -.024, .024, .044, .049]
    for xi in range(5):
        for yi in range(5):
            x0,x1,y0,y1=xs[xi],xs[xi+1],ys[yi],ys[yi+1]
            if xi not in (1,3) or yi not in (1,3):
                box(x0,x1,y0,y1,-.022,-.019)
                continue
            cx,cy=(x0+x1)/2,(y0+y1)/2
            radius=.006 if xi==1 else .00525
            for n in range(32):
                a,b=2*math.pi*n/32,2*math.pi*(n+1)/32
                def outer(t):
                    dx,dy=math.cos(t),math.sin(t)
                    scale=.010/max(abs(dx),abs(dy))
                    return cx+scale*dx,cy+scale*dy
                quad=[outer(a),outer(b),(cx+radius*math.cos(b),cy+radius*math.sin(b)),
                      (cx+radius*math.cos(a),cy+radius*math.sin(a))]
                points=[Gf.Vec3f(x,y,z) for z in (-.022,-.019) for x,y in quad]
                count += 1
                m=UsdGeom.Mesh.Define(stage,base+f'/Sector{count:03d}')
                m.CreatePointsAttr(points)
                m.CreateFaceVertexCountsAttr([4]*6)
                m.CreateFaceVertexIndicesAttr([3,2,1,0,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
                m.CreateSubdivisionSchemeAttr('none')
                UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
    for sign in (-1,1):
        def rail(y0,y1,z0,z1):
            box(-.075,.075,min(sign*y0,sign*y1),max(sign*y0,sign*y1),z0,z1)
        rail(.016,.019,-.019,.001)
        rail(.009,.019,-.002,.001)
        rail(.0145,.019,-.019,-.0175)
    return count


def add_bolt_shoulders(stage, paths, radius, lead_in=.002):
    """A short, visible bearing neck fits the keel holes, outside the wall.

    The 8 mm shaft still enters the 9 mm wall bore. This optional 10 mm
    neck bears on the 10.5/12 mm plate holes through ordinary collision.
    """
    if not radius:return
    for path in paths:
        # The right pair already proved stable with a full bearing neck.
        # Only the later left pair needs to guide the freely settled plate.
        if path.endswith(('Screw03','Screw04')):
            neck=UsdGeom.Cylinder.Define(stage,path+'/BearingShoulder')
            neck.CreateRadiusAttr(radius);neck.CreateHeightAttr(.003);neck.CreateAxisAttr('Z')
            neck.CreateDisplayColorAttr([Gf.Vec3f(.5,.53,.56)])
            UsdGeom.Xformable(neck).AddTranslateOp().Set(Gf.Vec3d(0,0,-.0019))
            UsdPhysics.CollisionAPI.Apply(neck.GetPrim()).CreateCollisionEnabledAttr(True)
            continue
        # A gradual lead-in guides a slightly settled plate onto the neck.
        # The full neck remains 3 mm long, fitting within
        # the plate. The leading end stays outside the wall at final depth.
        sections=((-0.0034,.004),(-.0034+lead_in,radius),(-.0004,radius))
        points=[Gf.Vec3f(r*math.cos(2*math.pi*i/32),r*math.sin(2*math.pi*i/32),z)
            for z,r in sections for i in range(32)]
        faces=[list(reversed(range(32))),list(range(64,96))]
        faces += [[level*32+i,level*32+(i+1)%32,(level+1)*32+(i+1)%32,(level+1)*32+i]
            for level in range(2) for i in range(32)]
        neck=UsdGeom.Mesh.Define(stage,path+'/BearingShoulder')
        neck.CreatePointsAttr(points);neck.CreateFaceVertexCountsAttr([len(f) for f in faces])
        neck.CreateFaceVertexIndicesAttr([i for f in faces for i in f]);neck.CreateSubdivisionSchemeAttr('none')
        neck.CreateDisplayColorAttr([Gf.Vec3f(.5,.53,.56)])
        UsdPhysics.CollisionAPI.Apply(neck.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(neck.GetPrim()).CreateApproximationAttr('convexHull')


def configure_contacts(stage, roots, keel_friction=(.4,.3)):
    material = UsdShade.Material.Define(stage,'/World/AssemblyMaterial')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(1.5)
    api.CreateDynamicFrictionAttr(1.2)
    api.CreateRestitutionAttr(0.)
    insertion_material=UsdShade.Material.Define(stage,'/World/AssemblyInsertionMaterial')
    api=UsdPhysics.MaterialAPI.Apply(insertion_material.GetPrim())
    api.CreateStaticFrictionAttr(.4);api.CreateDynamicFrictionAttr(.3);api.CreateRestitutionAttr(0.)
    keel_material=UsdShade.Material.Define(stage,'/World/KeelBearingMaterial')
    keel_api=UsdPhysics.MaterialAPI.Apply(keel_material.GetPrim())
    keel_api.CreateStaticFrictionAttr(keel_friction[0]);keel_api.CreateDynamicFrictionAttr(keel_friction[1])
    keel_api.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(keel_material.GetPrim()).CreateFrictionCombineModeAttr('max')
    insertion_colliders=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and any(str(prim.GetPath()).startswith(root) for root in roots):
            b=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            b.CreateSolverPositionIterationCountAttr(64)
            b.CreateSolverVelocityIterationCountAttr(8)
            b.CreateSleepThresholdAttr(0.)
            b.CreateMaxDepenetrationVelocityAttr(.10)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not any(str(prim.GetPath()).startswith(root) for root in roots):
            continue
        c=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        c.CreateContactOffsetAttr(.00015)
        c.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
            # The disabled original keel mesh is also the parent of its new
            # colliders. Allow the hole sectors to override its material.
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,materialPurpose='physics')
        path=str(prim.GetPath())
        if '/AssemblyWall/' in path or prim.GetName() in ('Shaft','BearingShoulder') or '/AssemblyCollision/Sector' in path:
            c.CreateContactOffsetAttr(.00005)
            selected=keel_material if '/AssemblyCollision/Sector' in path else insertion_material
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(selected,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
            insertion_colliders.append((prim,selected))
    # Verify resolved bindings, since an ancestor's stronger binding can
    # otherwise silently replace the lower-friction hole material.
    for prim,selected in insertion_colliders:
        resolved,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        if not resolved or resolved.GetPath()!=selected.GetPath():
            raise RuntimeError(f'Insertion material overridden at {prim.GetPath()}')
    return dict(insertion_colliders_checked=len(insertion_colliders),
        static_friction=float(api.GetStaticFrictionAttr().Get()),
        dynamic_friction=float(api.GetDynamicFrictionAttr().Get()),
        keel_static_friction=float(keel_api.GetStaticFrictionAttr().Get()),
        keel_dynamic_friction=float(keel_api.GetDynamicFrictionAttr().Get()))


def enable_dynamic_parts(stage, paths):
    """Parts remain dynamic and gravity-driven for the entire episode."""
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        body=UsdPhysics.RigidBodyAPI(prim)
        if not body:raise RuntimeError(f'Part is not a rigid body: {path}')
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False)


def verify_dynamic_parts(stage, paths):
    """Reject any joint connecting a part, including joints in custom scenes.

    Robot/base joints are independent of this check; no joint may constrain
    the keel or a screw to the wall, robot or another part.
    """
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        body=UsdPhysics.RigidBodyAPI(prim)
        if not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
            raise RuntimeError(f'Part must remain a dynamic rigid body: {path}')
        if PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get():
            raise RuntimeError(f'Gravity is disabled on part: {path}')
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim)
        targets=joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets()
        if any(str(target)==path or str(target).startswith(path+'/') for target in targets for path in paths):
            raise RuntimeError(f'Assembly part has a forbidden joint: {prim.GetPath()}')
    return dict(dynamic_parts=len(paths),gravity_enabled=True,assembly_joints=0)
