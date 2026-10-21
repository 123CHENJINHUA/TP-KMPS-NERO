"""Preinserted one/two-pin gravity bench; excluded from robot demonstrations."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from assembly_collection.geometry import pose, wall_targets, errors, insertion_metrics, keel_hole_metrics
from assembly_collection.physics import configure_contacts, preserve_keel_holes, add_bolt_shoulders, enable_dynamic_parts, verify_dynamic_parts
from assembly_collection.friction_bores import add_friction_bores


def collect(app,args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,rendering_dt=1/60,device='cpu')
    stage=omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(900.)
    add_reference_to_stage(str(args.scene_usd),'/World/UserScene')
    env='/World/UserScene/Environment';wall_path=env+'/AssemblyWall';keel_path=env+'/Geometry'
    for i in (1,2):stage.GetPrimAtPath(env+f'/Screw{i:02d}').SetActive(False)
    preserve_keel_holes(stage,keel_path)
    add_bolt_shoulders(stage,[env+f'/Screw{i:02d}' for i in (3,4)],args.bolt_shoulder_radius,args.bolt_shoulder_lead_in)
    for i in (3,4):
        head=UsdGeom.Cylinder(stage.GetPrimAtPath(env+f'/Screw{i:02d}/GripHead'))
        head.GetHeightAttr().Set(args.screw_head_height)
        next(o for o in UsdGeom.Xformable(head).GetOrderedXformOps()
            if o.GetOpType()==UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(0,0,args.screw_head_height/2))
    material=configure_contacts(stage,[env],(args.keel_static_friction,args.keel_dynamic_friction))
    contact_model=add_friction_bores(stage,wall_path,args)
    cache=UsdGeom.XformCache();matrix=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(wall_path))
    wall=pose(np.array(matrix.ExtractTranslation()),Rotation.from_matrix(np.array(matrix.ExtractRotationMatrix()).T))
    target,holes,bolts,normal=wall_targets(wall)
    set_camera_view(eye=np.array([.1,.5,1.6]),target=target[:3])
    keel=world.scene.add(SingleRigidPrim(keel_path,name='keel',reset_xform_properties=False))
    screws={i:world.scene.add(SingleRigidPrim(env+f'/Screw{i+1:02d}',name=f'bolt{i}',reset_xform_properties=False)) for i in (2,3)}
    parked=np.r_[screws[2].get_world_pose()[0],screws[2].get_world_pose()[1]]
    trials=[]
    for order in ((3,),(2,3)):
        world.stop()
        paths=[keel_path]+[env+f'/Screw{i+1:02d}' for i in (2,3)]
        enable_dynamic_parts(stage,paths)
        world.reset()
        # Bench initialization only. No pose commands or external forces
        # occur after this point; all subsequent movement is physical.
        start=target.copy();start[2]-=args.handoff_settle
        for obj,p in [(keel,start)]+[(screws[i],bolts[i] if i in order else parked) for i in (2,3)]:
            obj.set_world_pose(p[:3],p[3:]);obj.set_linear_velocity(np.zeros(3));obj.set_angular_velocity(np.zeros(3))
        rows=[]
        for tick in range(round(args.retention_seconds*args.physics_hz)):
            world.step(render=not args.headless)
            if tick%(args.physics_hz//args.record_hz)==0:
                p,q=keel.get_world_pose()
                rows.append(dict(time=(tick+1)/args.physics_hz,keel=np.r_[p,q],
                    screws=np.array([np.concatenate(screws[i].get_world_pose()) for i in order])))
        p,q=keel.get_world_pose();actual=np.r_[p,q]
        d,a=errors(actual,target)
        measurements=[]
        for i in order:
            p,q=screws[i].get_world_pose();screw=np.r_[p,q]
            m=insertion_metrics(screw,holes[i],normal);m.update(keel_hole_metrics(screw,actual,i,args.bolt_shoulder_radius or .004));m['screw_index']=i+1
            measurements.append(m)
        mounted=bool(d<.002 and a<.02 and all(.020<m['depth_m']<.0265 and m['axis_error_rad']<.05
            and m['keel_shaft_crosses_plate'] and m['keel_hole_radial_error_m']<m['keel_hole_clearance_m']+.0001 for m in measurements))
        result=dict(screw_order=[i+1 for i in order],keel_position_error_m=d,keel_angle_error_rad=a,
            vertical_drop_m=float(target[2]-actual[2]),final_keel_pose=actual.tolist(),screws=measurements,
            independently_mounted=mounted,physics_check=verify_dynamic_parts(stage,paths))
        trials.append(result);print('SUPPORT_PROBE '+json.dumps(result),flush=True)
        np.savez_compressed(args.output_dir/f'{len(order)}_pins.npz',times=np.array([r['time'] for r in rows]),
            keel_poses=np.array([r['keel'] for r in rows]),screw_poses=np.array([r['screws'] for r in rows]),
            task='keel_support_bench_v1',preinserted=True,robot_demonstration=False)
    summary=dict(task='keel_support_bench_v1',preinserted=True,robot_demonstration=False,
        settings={k:str(v) if hasattr(v,'is_file') else v for k,v in vars(args).items()},
        material_check=material,initial_bearing_drop_m=args.handoff_settle,
        bolt_shoulder_radius_m=args.bolt_shoulder_radius,
        contact_model={k:v for k,v in contact_model.items() if k!='collider_paths'},
        trials=trials,success=bool(not trials[0]['independently_mounted'] and trials[1]['independently_mounted']))
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.keep_open and not args.headless:
        while app.is_running():world.step(render=True)
    return 0 if summary['success'] else 1
