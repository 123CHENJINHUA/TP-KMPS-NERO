"""Preinserted bolt bench, not a robot insertion demonstration."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from assembly_collection.geometry import pose, wall_targets, insertion_metrics
from assembly_collection.physics import configure_contacts, enable_dynamic_parts, verify_dynamic_parts
from assembly_collection.friction_bores import add_friction_bores


def collect(app,args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,rendering_dt=1/60,device='cpu')
    stage=omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(900.)
    add_reference_to_stage(str(args.scene_usd),'/World/UserScene')
    env='/World/UserScene/Environment';wall_path=env+'/AssemblyWall';bolt_path=env+'/Screw04'
    for path in [env+'/Geometry']+[env+f'/Screw{i:02d}' for i in (1,2,3)]:
        stage.GetPrimAtPath(path).SetActive(False)
    head=UsdGeom.Cylinder(stage.GetPrimAtPath(bolt_path+'/GripHead'))
    head.GetHeightAttr().Set(args.screw_head_height)
    next(o for o in UsdGeom.Xformable(head).GetOrderedXformOps() if o.GetOpType()==UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(0,0,args.screw_head_height/2))
    configure_contacts(stage,[env])
    info=add_friction_bores(stage,wall_path,args)
    bolt=world.scene.add(SingleRigidPrim(bolt_path,name='bolt',reset_xform_properties=False))
    enable_dynamic_parts(stage,[bolt_path])
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(wall_path))
    wall=pose(np.array(matrix.ExtractTranslation()),Rotation.from_matrix(np.array(matrix.ExtractRotationMatrix()).T))
    _,holes,targets,normal=wall_targets(wall);target=targets[3]
    set_camera_view(eye=np.array([.1,.5,1.6]),target=holes[3])
    trials=[]
    # Include a friction-disabled control using the same contact geometry.
    material=UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/World/BoreFrictionMaterial'))
    for mu in (0.,args.bore_static_friction):
        world.stop()
        material.GetStaticFrictionAttr().Set(mu)
        material.GetDynamicFrictionAttr().Set(0. if mu==0 else args.bore_dynamic_friction)
        # With max combine, the control also needs a zero-friction shaft.
        shaft=stage.GetPrimAtPath(bolt_path+'/Shaft')
        from pxr import UsdShade
        control=UsdShade.Material.Define(stage,'/World/ProbeShaftMaterial')
        api=UsdPhysics.MaterialAPI.Apply(control.GetPrim())
        api.CreateStaticFrictionAttr(0.);api.CreateDynamicFrictionAttr(0.);api.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(shaft).Bind(control,bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
        world.reset()
        forces=RigidPrim(prim_paths_expr=bolt_path,name='probe_force',reset_xform_properties=False);forces.initialize()
        bolt.set_world_pose(target[:3],target[3:]);bolt.set_linear_velocity(np.zeros(3));bolt.set_angular_velocity(np.zeros(3))
        rows=[]
        for tick in range(round((args.retention_seconds+3.)*args.physics_hz)):
            seconds=tick/args.physics_hz
            # A finite 0.5 N outward disturbance distinguishes friction from
            # mere transverse resting contact. It is removed after one second.
            if 2.<=seconds<3.:forces.apply_forces(np.array([.5*normal],dtype=np.float32))
            world.step(render=not args.headless)
            if tick%(args.physics_hz//args.record_hz)==0:
                p,q=bolt.get_world_pose();rows.append(np.r_[seconds,p,q,bolt.get_linear_velocity()])
                # The zero-friction control is already disproved once it
                # leaves the bore. Avoid following a free-falling bolt far
                # outside the scene merely to fill the observation interval.
                if mu==0 and np.linalg.norm(p-target[:3])>.05:break
        p,q=bolt.get_world_pose();actual=np.r_[p,q]
        m=insertion_metrics(actual,holes[3],normal)
        delta=float(np.linalg.norm(actual[:3]-target[:3]))
        stable=(.022<m['depth_m']<.0265 and m['axis_error_rad']<.02 and delta<.0005)
        check=verify_dynamic_parts(stage,[bolt_path])
        trial=dict(static_friction=mu,stable=bool(stable),metrics=m,position_drift_m=delta,
            observation_seconds=(tick+1)/args.physics_hz,physics_check=check)
        trials.append(trial)
        label='control' if mu==0 else 'friction'
        np.savez_compressed(args.output_dir/f'{label}.npz',samples=np.array(rows),task='bolt_retention_bench_v1',preinserted=True)
        print('RETENTION_PROBE '+json.dumps(trial),flush=True)
    result=dict(task='bolt_retention_bench_v1',preinserted=True,robot_demonstration=False,
        settings={k:str(v) if hasattr(v,'is_file') else v for k,v in vars(args).items()},
        contact_model={k:v for k,v in info.items() if k!='collider_paths'},trials=trials,
        success=bool(trials[-1]['stable'] and not trials[0]['stable']))
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.keep_open and not args.headless:
        while app.is_running():world.step(render=True)
    return 0 if result['success'] else 1
