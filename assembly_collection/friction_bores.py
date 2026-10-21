"""Passive, finite-stiffness bore contact; no joints or pose/force servo."""
import math
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade


def add_friction_bores(stage, wall_path, args):
    """Two slightly undersized compliant rings per existing 9 mm wall hole.

    The rings are static wall geometry. Only contact compression produces
    pressure; friction is limited by that pressure and the material coefficient.
    A shaft removed from the rings experiences no retaining force.
    """
    material=UsdShade.Material.Define(stage,'/World/BoreFrictionMaterial')
    basic=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    basic.CreateStaticFrictionAttr(args.bore_static_friction)
    basic.CreateDynamicFrictionAttr(args.bore_dynamic_friction)
    basic.CreateRestitutionAttr(0.)
    physical=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physical.CreateFrictionCombineModeAttr('max')
    physical.CreateCompliantContactStiffnessAttr(args.bore_stiffness)
    physical.CreateCompliantContactDampingAttr(args.bore_damping)
    physical.CreateCompliantContactAccelerationSpringAttr(False)
    radius=.004-args.bore_interference
    base=wall_path+'/FrictionBores'
    UsdGeom.Xform.Define(stage,base)
    paths=[]
    for i,(x,z) in enumerate([(x,.1645+y) for x in (-.0615,.0615) for y in (-.034,.034)],1):
        for j,depth in enumerate((.008,.020),1):
            for n in range(32):
                a,b=2*math.pi*n/32,2*math.pi*(n+1)/32
                # Bevel both entrances: a square lip can load the flat bolt
                # tip axially instead of gently compressing the bore radially.
                points=[Gf.Vec3f(x+r*math.cos(t),-.015+depth+dy,z+r*math.sin(t))
                    for dy,inner in ((-.001,.0042),(0.,radius),(.001,.0042))
                    for r,t in ((.00449,a),(.00449,b),(inner,b),(inner,a))]
                vertices=np.asarray(points,dtype=float)
                hull=ConvexHull(vertices)
                triangles=[]
                for face,equation in zip(hull.simplices,hull.equations):
                    face=face.tolist()
                    if np.dot(np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]]),equation[:3])<0:
                        face[1],face[2]=face[2],face[1]
                    triangles.extend(face)
                mesh=UsdGeom.Mesh.Define(stage,base+f'/Hole{i}Ring{j}Sector{n:02d}')
                mesh.CreatePointsAttr(points)
                mesh.CreateFaceVertexCountsAttr([3]*len(hull.simplices))
                mesh.CreateFaceVertexIndicesAttr(triangles)
                mesh.CreateSubdivisionSchemeAttr('none')
                mesh.CreateDisplayColorAttr([Gf.Vec3f(.3,.32,.34)])
                prim=mesh.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(args.bolt_retention=='friction')
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(.00005)
                collision.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
                bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
                if bound.GetPath()!=material.GetPath():raise RuntimeError(f'Bore material overridden: {prim.GetPath()}')
                paths.append(str(prim.GetPath()))
    return dict(model='passive_compliant_bore_friction' if args.bolt_retention=='friction' else 'none',
        inner_diameter_m=2*radius,radial_interference_m=args.bore_interference,
        static_friction=args.bore_static_friction,dynamic_friction=args.bore_dynamic_friction,
        contact_stiffness_N_per_m=args.bore_stiffness,contact_damping_Ns_per_m=args.bore_damping,
        ring_depths_m=[.008,.020],ring_width_m=.002,lead_in_radius_m=.0042,collider_paths=paths)
