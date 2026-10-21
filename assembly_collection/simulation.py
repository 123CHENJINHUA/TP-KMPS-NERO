"""Independent dual-arm state machine and measured-state recorder."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import omni.kit.commands
import omni.usd
import omni.physx
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, PhysicsSchemaTools
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCuboid
from isaacsim.core.prims import Articulation, SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view

from assembly_collection.geometry import (compose, errors, interpolate, inverse,
    insertion_metrics, keel_hole_metrics, pose, rotation, top_grasp, keel_grasp, wall_targets, HOLES, screw_order, canonical_screw_target, axial_alignment_target)
from assembly_collection.physics import configure_contacts, preserve_keel_holes, add_bolt_shoulders, enable_dynamic_parts, verify_dynamic_parts
from assembly_collection.friction_bores import add_friction_bores
from sim_bridge.grasp_scene import configure_dual_grasp_contacts

ROOT=Path(__file__).resolve().parents[1]
JOINTS=[f'joint{i}' for i in range(1,8)]


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class TrialFailed(RuntimeError):
    pass


class AssemblySimulation:
    def __init__(self, app, args):
        self.app,self.args=app,args
        enable_extension('isaacsim.asset.importer.urdf')
        enable_extension('isaacsim.robot_motion.motion_generation')
        app.update()
        from isaacsim.asset.importer.urdf import _urdf
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
        self.world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,rendering_dt=1/60,device='cpu')
        self.stage=omni.usd.get_context().get_stage()
        self.world.scene.add(GroundPlane('/World/Ground',size=4.))
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath('/World'))
        UsdLux.DomeLight.Define(self.stage,'/World/Light').CreateIntensityAttr(900.)
        add_reference_to_stage(str(args.scene_usd),'/World/UserScene')
        self.environment='/World/UserScene/Environment'
        self.wall_path=self.environment+'/AssemblyWall'
        self.keel_path=self.environment+'/Geometry'
        for path in (self.wall_path,self.keel_path):
            if not self.stage.GetPrimAtPath(path).IsValid():
                raise TrialFailed(f'Scene lacks required prim: {path}')
        wall_xform=UsdGeom.Xformable(self.stage.GetPrimAtPath(self.wall_path))
        wall_translate=next(op for op in wall_xform.GetOrderedXformOps()
            if op.GetOpType()==UsdGeom.XformOp.TypeTranslate)
        wall_position=wall_translate.Get()
        wall_translate.Set(Gf.Vec3d(wall_position[0]+args.wall_offset_x,wall_position[1],wall_position[2]))
        rack=UsdGeom.Xformable(self.stage.GetPrimAtPath(self.environment+'/ScrewRack'))
        rack.AddTranslateOp(opSuffix='assemblyClearance').Set(Gf.Vec3d(0,args.rack_offset_y,0))
        for i in range(1,5):
            screw=UsdGeom.Xformable(self.stage.GetPrimAtPath(self.environment+f'/Screw{i:02d}'))
            op=next(o for o in screw.GetOrderedXformOps() if o.GetOpType()==UsdGeom.XformOp.TypeTranslate)
            p=op.Get();op.Set(Gf.Vec3d(p[0],p[1]+args.rack_offset_y,p[2]))
        # The original 14 mm heads put the wide finger tips into the keel ribs
        # before seating. Extend only the live head, preserving shaft and seat.
        for i in range(1,5):
            head=UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.environment+f'/Screw{i:02d}/GripHead'))
            head.GetHeightAttr().Set(args.screw_head_height)
            op=next(o for o in UsdGeom.Xformable(head).GetOrderedXformOps()
                if o.GetOpType()==UsdGeom.XformOp.TypeTranslate)
            op.Set(Gf.Vec3d(0,0,args.screw_head_height/2))
        preserve_keel_holes(self.stage,self.keel_path)
        self.part_paths=[self.keel_path]+[self.environment+f'/Screw{i:02d}' for i in range(1,5)]
        add_bolt_shoulders(self.stage,self.part_paths[1:],args.bolt_shoulder_radius,args.bolt_shoulder_lead_in)
        enable_dynamic_parts(self.stage,self.part_paths)
        self.arms=[]
        for i,y in enumerate((-.375,.375),1):
            self.world.scene.add(FixedCuboid(f'/World/Stand{i}',name=f'stand{i}',
                position=np.array([0,y,.5]),scale=np.array([.24,.24,1.]),color=np.array([.22,.27,.32])))
            status,config=omni.kit.commands.execute('URDFCreateImportConfig')
            config.merge_fixed_joints=False; config.fix_base=True; config.import_inertia_tensor=True
            config.distance_scale=1.; config.self_collision=False
            config.default_drive_strength=20000.; config.default_position_drive_damping=600.
            config.default_drive_type=_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
            urdf=ROOT/'assets/nero'/('nero.urdf' if i==1 else 'nero_dual_2.urdf')
            self.stage.SetDefaultPrim(self.stage.GetPrimAtPath('/World'))
            status,path=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=str(urdf),
                import_config=config,get_articulation_root=True)
            if not status: raise TrialFailed(f'URDF import failed for arm {i}')
            root=path.rsplit('/',1)[0]
            t=UsdGeom.Xformable(self.stage.GetPrimAtPath(root))
            ops={o.GetOpType():o for o in t.GetOrderedXformOps()}
            (ops.get(UsdGeom.XformOp.TypeTranslate) or t.AddTranslateOp()).Set(Gf.Vec3d(0,y,1.))
            robot=self.world.scene.add(SingleArticulation(prim_path=path,name=f'arm{i}'))
            marker=self.world.scene.add(VisualCuboid(f'/World/Goal{i}',name=f'goal{i}',size=.012,
                color=np.array([.9,.2,.1] if i==1 else [.1,.3,.9])))
            solver=LulaKinematicsSolver(str(ROOT/'assets/lula.yaml'),str(urdf))
            solver.set_robot_base_pose(np.array([0,y,1.]),np.array([1.,0,0,0]))
            self.arms.append(dict(robot=robot,root=root,path=path,solver=solver,marker=marker,opening=.06))
        configure_dual_grasp_contacts(self.world,[a['root'] for a in self.arms])
        self.material_check=configure_contacts(self.stage,[self.environment]+[a['root'] for a in self.arms],
            (args.keel_static_friction,args.keel_dynamic_friction))
        self.retention_model=(add_friction_bores(self.stage,self.wall_path,args) if args.bolt_retention=='friction'
            else dict(model='none'))
        # Detect contact between the arms rather than letting the holding arm
        # compensate for a collision and accidentally push the keel out of grip.
        self.cross_collision=None;self.recent_contacts={}
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and any(
                    str(prim.GetPath()).startswith(a['root']+'/') for a in self.arms):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(.1)
        def on_contacts(headers, data):
            for h in headers:
                p0=str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                p1=str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                r0=next((i for i,a in enumerate(self.arms) if p0.startswith(a['root']+'/')),None)
                r1=next((i for i,a in enumerate(self.arms) if p1.startswith(a['root']+'/')),None)
                if h.num_contact_data and (r0 is not None or r1 is not None):
                    self.recent_contacts[(p0,p1)]=getattr(self,'tick',0)
                if r0 is not None and r1 is not None and r0!=r1 and h.num_contact_data:
                    self.cross_collision=f'{p0} <-> {p1}'
        self.contact_subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        self.keel=self.world.scene.add(SingleRigidPrim(self.keel_path,name='keel',reset_xform_properties=False))
        self.screws=[self.world.scene.add(SingleRigidPrim(self.environment+f'/Screw{i:02d}',
            name=f'screw{i}',reset_xform_properties=False)) for i in range(1,5)]
        self.initial_objects=[self.object_pose(o) for o in [self.keel]+self.screws]
        # Real pickup supports provide floor clearance for a tilted grasp.
        # They act only through collision and never attach to the keel.
        if args.pickup_rest_height:
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
            table_top=float(cache.ComputeWorldBound(self.stage.GetPrimAtPath(self.environment+'/Cube')).ComputeAlignedRange().GetMax()[2])
            start=self.initial_objects[0].copy();start[0]+=args.keel_offset_x
            rz=Rotation.from_euler('z',np.deg2rad(args.keel_start_yaw))*rotation(start)
            for i,x in enumerate((-.052,.052)):
                center=start[:3]+rz.apply([x,0,0]);center[2]=table_top+args.pickup_rest_height/2
                self.world.scene.add(FixedCuboid(f'/World/KeelPickupRest{i}',name=f'keel_rest{i}',
                    position=center,orientation=rz.as_quat()[[3,0,1,2]],scale=np.array([.016,.028,args.pickup_rest_height]),
                    color=np.array([.27,.33,.38])))
        self.keel_grip=np.array([args.left_grip_x,0.,0.])
        self.world.reset()
        for i,a in enumerate(self.arms):
            r=a['robot']; names=r.dof_names
            a['indices']=np.array([names.index(n) for n in JOINTS])
            a['fingers']=np.array([names.index(n) for n in ('gripper_joint1','gripper_joint2')])
            a['kinematics']=ArticulationKinematicsSolver(r,a['solver'],'gripper_base')
            a['dynamics']=Articulation(prim_paths_expr=a['path'],name=f'dynamics{i}')
            a['dynamics'].initialize()
            kp=np.full(r.num_dof,20000.);kd=np.full(r.num_dof,600.)
            kp[a['fingers']]=args.finger_stiffness if i==1 else 600.;kd[a['fingers']]=10.
            r.get_articulation_controller().set_gains(kps=kp,kds=kd)
            r.get_articulation_controller().set_max_efforts(np.full(2,10.),joint_indices=a['fingers'])
            r.set_solver_position_iteration_count(64)
            a['command']=r.get_joint_positions().copy()
            a['desired']=self.tcp(i)
        self.wall=self.prim_pose(self.wall_path)
        self.keel_target,self.holes,self.screw_targets,self.normal=wall_targets(self.wall)
        for i,target in enumerate(self.screw_targets):
            clock=np.sign(HOLES[i,0]*HOLES[i,1])*np.deg2rad(args.screw_clock_deg)
            target[3:]=(rotation(target)*Rotation.from_euler('z',clock)).as_quat()[[3,0,1,2]]
        self.nominal_screw_targets=self.screw_targets.copy()
        self.keel_support_target=self.keel_target.copy()
        set_camera_view(eye=np.array([1.1,1.2,1.85]),target=np.array([-.43,0,1.1]))
        self.tick=0;self.rows=[];self.labels=[];self.label='initialization';self.active=-1
        self.hold_keel=False;self.record=False
        self.hold_screw=None
        self.inserted=np.zeros(4,dtype=bool)
        self.grasp_checks=[];self.insertion_events=[];self.retention_checks=[];self.release_observation=None
        self.left_release_time=-1.
        self.physics_check=verify_dynamic_parts(self.stage,self.part_paths)
        print('ASSEMBLY_PHYSICS '+json.dumps(self.physics_check),flush=True)
        print('ASSEMBLY_MATERIAL '+json.dumps(self.material_check),flush=True)
        print('ASSEMBLY_RETENTION_MODEL '+json.dumps({k:v for k,v in self.retention_model.items() if k!='collider_paths'}),flush=True)

    def prim_pose(self,path):
        t=UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
        return pose(np.array(t.ExtractTranslation()),Rotation.from_matrix(np.array(t.ExtractRotationMatrix()).T))

    @staticmethod
    def object_pose(o):
        p,q=o.get_world_pose();return np.r_[p,q]

    def tcp(self,index):
        p,r=self.arms[index]['kinematics'].compute_end_effector_pose()
        return pose(p,Rotation.from_matrix(r))

    def solve(self,index,wanted,seed=None,retry=False):
        a=self.arms[index]
        seed=a['robot'].get_joint_positions()[a['indices']] if seed is None else seed
        seeds=[seed]
        if retry:
            seeds += [np.array([2.1,.28,1.26,1.67,-.22,.27,1.38]),
                      np.array([-2.1,.28,-1.26,1.67,.22,.27,-1.38]),
                      np.array([0.,-.4,0.,1.,0.,.2,0.])]
            seeds += list(np.random.default_rng(42).uniform([-2.7,-1.7,-2.7,-1.,-2.7,-.7,-1.5],
                [2.7,1.7,2.7,2.1,2.7,.9,1.5],size=(24,7)))
        candidates=[]
        for start in seeds:
            q,ok=a['solver'].compute_inverse_kinematics('gripper_base',wanted[:3],wanted[3:],
                warm_start=start,position_tolerance=.00015,orientation_tolerance=.003)
            if ok and np.isfinite(q).all():
                if not retry or self.args.elbow_policy=='first':return q
                elbow,_=a['solver'].compute_forward_kinematics('link4',q)
                # Select only the initialization branch. Subsequent IK stays
                # warm-started so there is no discontinuous posture switch.
                score=-elbow[2] if index==0 else abs(elbow[2]-1.12)
                candidates.append((score,q))
        if candidates:return min(candidates,key=lambda entry:entry[0])[1]
        raise TrialFailed(f'IK unreachable: arm={index+1}, stage={self.label}, pose={wanted.tolist()}')

    def command_tcp(self,index,wanted):
        a=self.arms[index]
        seed=a.get('posture_seed')
        cached=a.get('ik_goal')
        if seed is None and cached is not None and np.array_equal(cached[0],wanted):
            q=cached[1]
        else:
            if seed is None:seed=a['robot'].get_joint_positions()[a['indices']]
            q=self.solve(index,wanted,seed=seed)
            # Near a joint limit Lula can choose different valid redundant
            # solutions on successive calls. Rate limiting each changing
            # solution can prevent convergence even with no contact. Retain
            # one joint solution while the Cartesian target is unchanged.
            a['ik_goal']=(wanted.copy(),q.copy())
        ix=a['indices'];a['command'][ix] += np.clip(q-a['command'][ix],-.8/self.args.physics_hz,.8/self.args.physics_hz)
        a['desired']=wanted.copy()
        a['marker'].set_world_pose(wanted[:3],wanted[3:])

    def adjust_support_posture(self):
        """Move in the IK null space, retaining the commanded hand pose.

        This is an optional geometric posture preference, not collision
        avoidance: real contact reports still reject any arm collision.
        """
        a=self.arms[0];solver=a['solver']
        q=a['robot'].get_joint_positions()[a['indices']]
        p,r=solver.compute_forward_kinematics('gripper_base',q)
        elbow,_=solver.compute_forward_kinematics('link4',q)
        jacobian=np.zeros((6,7));gradient=np.zeros(7);epsilon=1e-4
        for j in range(7):
            shifted=q.copy();shifted[j]+=epsilon
            p1,r1=solver.compute_forward_kinematics('gripper_base',shifted)
            e1,_=solver.compute_forward_kinematics('link4',shifted)
            jacobian[:3,j]=(p1-p)/epsilon
            jacobian[3:,j]=Rotation.from_matrix(r1@r.T).as_rotvec()/epsilon
            gradient[j]=(e1[2]-elbow[2])/epsilon
        projector=np.eye(7)-np.linalg.pinv(jacobian)@jacobian
        direction=projector@gradient
        step=np.clip((self.args.left_elbow_height-elbow[2])/max(np.dot(gradient,direction),1e-8),-3.,3.)*direction
        norm=np.linalg.norm(step)
        if norm>.025:step*=.025/norm
        a['posture_seed']=q+step

    def step(self):
        if not self.app.is_running(): raise TrialFailed('Simulation window closed')
        if not self.world.is_playing(): raise TrialFailed('Timeline paused or stopped during automatic collection')
        if self.cross_collision:raise TrialFailed('Arm collision: '+self.cross_collision)
        if self.hold_keel:
            if self.tick % (self.args.physics_hz//10)==0:
                if self.args.left_elbow_height:self.adjust_support_posture()
                current=self.object_pose(self.keel)
                previous=self.arms[0]['desired']
                relative=compose(inverse(current),previous)
                self.keel_servo_target=interpolate(previous,compose(self.keel_support_target,relative),.30)
            self.command_tcp(0,self.keel_servo_target)
        if self.hold_screw is not None:
            obj,target=self.hold_screw
            if self.tick % (self.args.physics_hz//30)==0:
                self.screw_hold_target=self.screw_feedback_target(obj,target)
            self.command_tcp(1,self.screw_hold_target)
        for a in self.arms:
            a['robot'].set_joint_efforts(a['dynamics'].get_generalized_gravity_forces()[0])
            goal=np.array([a['opening']/2,-a['opening']/2])
            ix=a['fingers'];a['command'][ix] += np.clip(goal-a['command'][ix],-.025/self.args.physics_hz,.025/self.args.physics_hz)
            a['robot'].apply_action(ArticulationAction(joint_positions=a['command'].copy()))
        self.world.step(render=not self.args.headless)
        self.tick += 1
        if self.record and self.tick % (self.args.physics_hz//self.args.record_hz)==0:
            self.snapshot()

    def screw_feedback_target(self,obj,wanted):
        previous=self.arms[1]['desired']
        lag,angle=errors(self.tcp(1),previous)
        # Retain the calibrated drive command, but wait for the hand to
        # follow it before accumulating another object-pose correction.
        # Replacing the command with actual TCP would discard the drive
        # deflection needed to carry the load even outside contact.
        if lag>.006 or angle>.06:return previous.copy()
        current=self.object_pose(obj)
        aligned=axial_alignment_target(wanted,current)
        correction=compose(aligned,compose(inverse(current),previous))
        return interpolate(previous,correction,.5)

    def snapshot(self):
        if self.label not in self.labels:self.labels.append(self.label)
        self.rows.append(dict(time=self.tick/self.args.physics_hz,
            tcp=np.array([self.tcp(i) for i in range(2)]),
            desired=np.array([a['desired'] for a in self.arms]),
            joints=np.array([a['robot'].get_joint_positions()[a['indices']] for a in self.arms]),
            joint_targets=np.array([a['command'][a['indices']] for a in self.arms]),
            fingers=np.array([a['robot'].get_joint_positions()[a['fingers']] for a in self.arms]),
            opening=np.array([a['opening'] for a in self.arms]),
            keel=self.object_pose(self.keel),screws=np.array([self.object_pose(o) for o in self.screws]),
            inserted=self.inserted.copy(),support=self.hold_keel,
            keel_velocity=self.keel.get_linear_velocity(),screw_velocities=np.array([o.get_linear_velocity() for o in self.screws]),
            stage=self.labels.index(self.label),screw=self.active))

    def wait(self,seconds,label=None):
        if label:self.set_stage(label)
        for _ in range(round(seconds*self.args.physics_hz)):self.step()

    def open_at_measured_pose(self,index,label):
        # Remove the loaded pose-controller bias while opening. Otherwise
        # the wrist can rebound toward that bias and pull the bolt outward
        # while the fingers still touch it. Only robot targets are changed.
        wanted=self.tcp(index)
        self.arms[index]['opening']=.07
        self.set_stage(label)
        for _ in range(round(1.5*self.args.physics_hz)):
            self.command_tcp(index,wanted);self.step()

    def set_stage(self,label):
        self.label=label
        print(f'ASSEMBLY_STAGE {label}',flush=True)

    def move(self,index,wanted,label,speed=None):
        self.set_stage(label)
        start=self.tcp(index)
        dist,angle=errors(start,wanted)
        duration=max(1.0,dist/(speed or self.args.speed),angle/.30)
        count=round(duration*self.args.physics_hz)
        seed=self.arms[index]['robot'].get_joint_positions()[self.arms[index]['indices']]
        # Solve the entire segment before moving; avoid knowingly entering an unreachable path.
        for fraction in np.linspace(0,1,25):seed=self.solve(index,interpolate(start,wanted,fraction),seed=seed)
        for tick in range(1,count+1):
            self.command_tcp(index,interpolate(start,wanted,tick/count));self.step()
        for _ in range(self.args.physics_hz*8):
            self.command_tcp(index,wanted);self.step()
            dist,angle=errors(self.tcp(index),wanted)
            # Transport can deflect a loaded tilted wrist slightly. Actual
            # object/hole alignment uses its own stricter checks below.
            if dist<.001 and angle<.02:return
        contacts=[pair for pair,tick in self.recent_contacts.items() if tick>self.tick-self.args.physics_hz]
        raise TrialFailed(f'{label}: TCP tracking failed, position={dist:.4f}m angle={angle:.4f}rad, contacts={contacts}')

    def initialize(self,variation):
        self.screw_targets=self.nominal_screw_targets.copy()
        self.keel_support_target=self.keel_target.copy()
        self.record=False;self.hold_keel=False
        self.hold_screw=None
        self.tick=0;self.rows=[];self.labels=[];self.label='initialization';self.active=-1
        self.cross_collision=None;self.recent_contacts={};self.screw_grip_points={}
        self.inserted=np.zeros(4,dtype=bool)
        self.grasp_checks=[];self.insertion_events=[];self.retention_checks=[];self.release_observation=None
        self.left_release_time=-1.
        self.world.stop()
        self.world.reset()
        self.physics_check=verify_dynamic_parts(self.stage,self.part_paths)
        for i,a in enumerate(self.arms):
            a.pop('posture_seed',None)
            a.pop('ik_goal',None)
            a['dynamics']=Articulation(prim_paths_expr=a['path'],name=f'dynamics{i}',reset_xform_properties=False)
            a['dynamics'].initialize()
            r=a['robot'];kp=np.full(r.num_dof,20000.);kd=np.full(r.num_dof,600.)
            kp[a['fingers']]=self.args.finger_stiffness if i==1 else 600.;kd[a['fingers']]=10.
            r.get_articulation_controller().set_gains(kps=kp,kds=kd)
            r.get_articulation_controller().set_max_efforts(np.full(2,10.),joint_indices=a['fingers'])
        for o,p in zip([self.keel]+self.screws,self.initial_objects):
            p=p.copy()
            if o is self.keel:
                p[:2] += variation
                p[0] += self.args.keel_offset_x
                p[2] += self.args.pickup_rest_height
                p[3:]=(Rotation.from_euler('z',np.deg2rad(self.args.keel_start_yaw))*rotation(p)).as_quat()[[3,0,1,2]]
            o.set_world_pose(p[:3],p[3:]);o.set_linear_velocity(np.zeros(3));o.set_angular_velocity(np.zeros(3))
        wanted=keel_grasp(self.object_pose(self.keel),self.args.left_grip_x,np.deg2rad(self.args.left_tilt_deg));wanted[2]+=.020
        first=self.args.first_screw-1
        parked=top_grasp(self.object_pose(self.screws[first]),
            (0,np.sign(HOLES[first,0])*self.args.screw_grip_offset,self.args.screw_head_height-.007),np.pi/2)
        parked[2]+=.045
        for index,target in enumerate((wanted,parked)):
            a=self.arms[index];r=a['robot'];q=self.solve(index,target,retry=True)
            full=np.zeros(r.num_dof);full[a['indices']]=q;full[a['fingers']]=[.035,-.035]
            r.set_joint_positions(full);r.set_joint_velocities(np.zeros(r.num_dof))
            a['command']=full.copy();a['opening']=.07;a['desired']=target.copy()
            r.apply_action(ArticulationAction(joint_positions=full))
        self.tick=0;self.rows=[];self.labels=[];self.label='initialization';self.active=-1
        self.wait(1.5)
        self.keel_start=self.object_pose(self.keel)
        print(f'ASSEMBLY_INITIAL keel={self.keel_start.tolist()} wall_target={self.keel_target.tolist()}',flush=True)
        self.record=True
        self.snapshot()

    def verify_grasp(self,index,obj,start,label):
        actual=self.object_pose(obj);r=self.arms[index]['robot'];f=self.arms[index]['fingers']
        gap=float(r.get_joint_positions()[f][0]-r.get_joint_positions()[f][1])
        lift=float(actual[2]-start[2]);center=self.tcp(index)[:3]+rotation(self.tcp(index)).apply([0,0,.125])
        grip_point=self.keel_grip if obj is self.keel else self.screw_grip_points[self.active]
        actual_grip_point=actual[:3]+rotation(actual).apply(grip_point)
        distance=float(np.linalg.norm(actual_grip_point-center))
        expected_gap=.038 if obj is self.keel else .018
        samples=[row for row in self.rows if row['time']>self.tick/self.args.physics_hz-.45]
        relatives=[compose(inverse(row['tcp'][index]),row['keel'] if obj is self.keel else row['screws'][self.active])[:3] for row in samples]
        slip=float(np.max(np.linalg.norm(np.array(relatives)-relatives[0],axis=1))) if relatives else float('inf')
        if lift<.045 or not .6*expected_gap<gap<1.4*expected_gap or distance>.012 or slip>.003:
            raise TrialFailed(f'{label}: physical grasp failed, lift={lift:.4f}, gap={gap:.4f}, center_error={distance:.4f}')
        self.grasp_checks.append(dict(label=label,arm=index+1,time_s=self.tick/self.args.physics_hz,
            lift_m=lift,gap_m=gap,grip_point_error_m=distance,relative_slip_m=slip))
        print(f'ASSEMBLY_CHECK {label} lift={lift:.4f} gap={gap:.4f}',flush=True)
        return compose(inverse(actual),self.tcp(index))

    def grab_keel(self):
        start=self.object_pose(self.keel)
        target=keel_grasp(start,self.args.left_grip_x,np.deg2rad(self.args.left_tilt_deg))
        self.move(0,target,'keel_descend')
        self.arms[0]['opening']=0.
        self.wait(2.,'keel_close')
        lifted=target.copy();lifted[:3]+=[.095-self.args.keel_offset_x,-.08,.115]
        self.move(0,lifted,'keel_lift')
        self.wait(.75,'keel_grasp_hold')
        self.keel_relative=self.verify_grasp(0,self.keel,start,'keel_grasp')

    def align_keel(self):
        current=self.object_pose(self.keel)
        rotated=current.copy();rotated[3:]=self.keel_target[3:]
        self.move(0,compose(rotated,self.keel_relative),'keel_rotate')
        approach=self.keel_target.copy();approach[:3] += .045*self.normal
        self.move(0,compose(approach,self.keel_relative),'keel_wall_approach')
        self.keel_relative=self.refine_object(0,self.keel,approach,'keel_pre_align')
        self.move(0,compose(self.keel_target,self.keel_relative),'keel_press',speed=.008)
        self.keel_servo_target=self.tcp(0)
        self.hold_keel=True
        self.wait(5.,'keel_align_hold')
        d,a=errors(self.object_pose(self.keel),self.keel_target)
        if d>.001 or a>.01:
            raise TrialFailed(f'Keel wall alignment failed: {d:.5f}m {a:.5f}rad; actual_joints={self.arms[0]["robot"].get_joint_positions().tolist()}; command={self.arms[0]["command"].tolist()}')
        print(f'ASSEMBLY_CHECK keel_aligned error={d:.6f}m angle={a:.6f}',flush=True)
        actual=self.object_pose(self.keel)
        centers=actual[:3]+rotation(actual).apply(HOLES)
        delta=centers-self.holes
        axial=delta@self.normal
        radial=np.linalg.norm(delta-axial[:,None]*self.normal,axis=1)
        if radial.max()>.00035:
            raise TrialFailed(f'Keel hole alignment exceeds insertion clearance: {radial.tolist()}')
        self.alignment_check=dict(position_error_m=d,angle_error_rad=a,hole_radial_errors_m=radial.tolist())

    def refine_object(self,index,obj,target,label,axis_only=False):
        for attempt in range(8):
            current=self.object_pose(obj)
            relative=compose(inverse(current),self.arms[index]['desired'])
            aligned=axial_alignment_target(target,current) if axis_only else target
            self.move(index,compose(aligned,relative),f'{label}_{attempt+1}',speed=.008)
            self.wait(.4)
            actual=self.object_pose(obj)
            d,a=errors(actual,target)
            if axis_only:
                m=insertion_metrics(actual,self.holes[self.active],self.normal)
                a=m['axis_error_rad']
                precise=(d<.0003 and a<.004 and max(m['radial_error_m'],m['entry_radial_error_m'])<.00035)
            else:precise=d<.0003 and a<.003
            print(f'ASSEMBLY_REFINE {label} error={d:.6f} angle={a:.6f}',flush=True)
            if precise:
                return compose(inverse(actual),self.arms[index]['desired'])
        raise TrialFailed(f'{label}: measured object alignment failed: {d:.6f}m {a:.6f}rad')

    def grab_screw(self,index):
        self.active=index
        obj=self.screws[index];start=self.object_pose(obj)
        # Centre the cylindrical head between the finger pads. Its axial roll
        # in the rack must not change the world-frame approach direction.
        world_offset=np.array([0.,np.sign(HOLES[index,0])*self.args.screw_grip_offset,self.args.screw_head_height-.007])
        self.screw_grip_points[index]=rotation(start).inv().apply(world_offset)
        upright=pose(start[:3],Rotation.identity())
        target=top_grasp(upright,local_point=world_offset,yaw=np.pi/2)
        approach=target.copy();approach[2]+=.060
        self.arms[1]['opening']=.06
        self.move(1,approach,f'screw{index+1}_approach')
        self.move(1,target,f'screw{index+1}_descend',speed=.015)
        self.arms[1]['opening']=0.
        self.wait(1.8,f'screw{index+1}_close')
        lifted=target.copy();lifted[2]+=.080
        self.move(1,lifted,f'screw{index+1}_lift')
        self.wait(.5,f'screw{index+1}_grasp_hold')
        return self.verify_grasp(1,obj,start,f'screw{index+1}_grasp')

    def insert_screw(self,index,relative):
        obj=self.screws[index];current=self.object_pose(obj)
        self.screw_targets[index]=canonical_screw_target(self.nominal_screw_targets[index],relative)
        tool_rotation=rotation(self.screw_targets[index])*rotation(relative)
        rotation_target=pose(np.array([-.10,.0795,1.40]),tool_rotation)
        self.move(1,rotation_target,f'screw{index+1}_rotate')
        target=self.screw_targets[index]
        approach=target.copy();approach[:3] += self.args.approach_distance*self.normal
        approach_tcp=compose(approach,relative)
        transit=rotation_target.copy();transit[1]=approach_tcp[1]
        self.move(1,transit,f'screw{index+1}_hole_transit')
        lower_route=self.args.lower_hole_route=='below' and HOLES[index,1]<0
        if lower_route:
            outside=transit.copy();outside[0]=.03
            self.move(1,outside,f'screw{index+1}_outside_support')
            lower=outside.copy();lower[2]=approach_tcp[2]
            self.move(1,lower,f'screw{index+1}_below_support')
        else:
            above=approach_tcp.copy();above[2]=rotation_target[2]
            self.move(1,above,f'screw{index+1}_above_hole')
        self.move(1,approach_tcp,f'screw{index+1}_hole_approach')
        relative=self.refine_object(1,obj,approach,f'screw{index+1}_pre_align',axis_only=True)
        self.insert_follow(index,obj,approach,target)
        self.hold_screw=(obj,target)
        self.screw_hold_target=self.arms[1]['desired'].copy()
        self.wait(2.,f'screw{index+1}_insert_hold')
        actual=self.object_pose(obj)
        metrics=insertion_metrics(actual,self.holes[index],self.normal)
        metrics.update(keel_hole_metrics(actual,self.object_pose(self.keel),index,self.args.bolt_shoulder_radius or .004))
        if not (.022<metrics['depth_m']<.0265 and max(metrics['radial_error_m'],metrics['entry_radial_error_m'])<.00045 and metrics['axis_error_rad']<.02):
            raise TrialFailed(f'Screw {index+1} insertion rejected: {metrics}')
        margin=.00005 if self.args.bolt_shoulder_radius else .00015
        limit=metrics['keel_hole_clearance_m']-margin if self.hold_keel else metrics['keel_hole_clearance_m']+.0001
        if not metrics['keel_shaft_crosses_plate'] or metrics['keel_hole_radial_error_m']>=limit:
            raise TrialFailed(f'Screw {index+1} does not pass through the actual keel hole: {metrics}')
        metrics['left_support_active_at_insertion']=self.hold_keel
        self.set_stage(f'screw{index+1}_insertion_verified')
        self.insertion_events.append(dict(time_s=self.tick/self.args.physics_hz,
            kind='geometric_insertion',screw_index=index+1,pose=actual.tolist(),metrics=metrics.copy()))
        self.hold_screw=None
        self.open_at_measured_pose(1,f'screw{index+1}_release')
        retreat=self.tcp(1);retreat[:3]+=.035*self.normal
        self.move(1,retreat,f'screw{index+1}_retreat')
        if lower_route:
            outside=retreat.copy();outside[0]=.03
            self.move(1,outside,f'screw{index+1}_clear_below')
            raised=outside.copy()
        else:raised=retreat.copy()
        raised[2]=rotation_target[2]
        self.move(1,raised,f'screw{index+1}_raise_clear')
        clear=rotation_target.copy()
        self.move(1,clear,f'screw{index+1}_clear')
        self.set_stage(f'screw{index+1}_retention_observation')
        observations=[]
        for _ in range(round(self.args.retention_seconds*self.args.physics_hz)):
            self.step()
            observations.append(self.object_pose(obj)[:3])
        released=self.contact_status(index)
        path=self.environment+f'/Screw{index+1:02d}'
        hand_contacts=[pair for pair,tick in self.recent_contacts.items() if tick>self.tick-self.args.physics_hz and path in pair]
        arm=self.arms[1]
        gap=float(arm['robot'].get_joint_positions()[arm['fingers']][0]-arm['robot'].get_joint_positions()[arm['fingers']][1])
        instantaneous_speed=float(np.linalg.norm(obj.get_linear_velocity()))
        window=np.array(observations[-self.args.physics_hz:])
        window_seconds=(len(window)-1)/self.args.physics_hz
        speed=float(np.linalg.norm(window[-1]-window[0])/window_seconds)
        excursion=float(np.max(np.linalg.norm(window-window[0],axis=1)))
        depth_loss=float(metrics['depth_m']-released['depth_m'])
        stable=bool(released['retained'] and not hand_contacts and gap>.060 and speed<.001
            and excursion<.00025 and abs(depth_loss)<.001)
        self.retention_checks.append(dict(screw_index=index+1,time_s=self.tick/self.args.physics_hz,
            observation_seconds=self.args.retention_seconds,verified=stable,
            depth_loss_m=depth_loss,speed_m_per_s=speed,instantaneous_speed_m_per_s=instantaneous_speed,
            position_excursion_m=excursion,stability_window_seconds=window_seconds,
            gripper_gap_m=gap,hand_contacts=hand_contacts,metrics=released))
        if not stable and self.args.stage!='release-test':
            raise TrialFailed(f'Screw {index+1} free retention check failed: {self.retention_checks[-1]}')
        if not released['retained'] and self.args.stage!='release-test':
            raise TrialFailed(f'Screw {index+1} did not remain inserted after physical release: {released}')
        self.inserted[index]=released['retained']
        print(f'ASSEMBLY_CHECK screw{index+1}_contact_retained {released}',flush=True)
        metrics['screw_index']=index+1
        metrics['released_metrics']=released
        return metrics

    def insert_follow(self,index,obj,start,target):
        self.set_stage(f'screw{index+1}_insert')
        # A compressed contact ring can elastically resist each advance.
        # Allow bounded lag inside the same final axis tolerance; the final
        # depth, centerline and actual keel passage checks remain mandatory.
        axial_tolerance=.0005 if self.args.bolt_retention=='friction' else .0002
        axis_tolerance=.02 if self.args.bolt_retention=='friction' else .004
        count=int(np.ceil(np.linalg.norm(target[:3]-start[:3])/(self.args.insert_speed*.10)))
        for number,fraction in enumerate(np.linspace(0,1,count+1)[1:],1):
            wanted=target.copy();wanted[:3]=(1-fraction)*start[:3]+fraction*target[:3]
            for attempt in range(self.args.physics_hz*3):
                if attempt % (self.args.physics_hz//30)==0:
                    self.screw_servo_target=self.screw_feedback_target(obj,wanted)
                self.command_tcp(1,self.screw_servo_target)
                self.step()
                actual=self.object_pose(obj)
                d,a=errors(actual,wanted)
                m=insertion_metrics(actual,self.holes[index],self.normal)
                axial_error=abs(float(np.dot(actual[:3]-wanted[:3],self.normal)))
                # A cylinder's axial roll is unconstrained by a round hole.
                # Require its entire centerline to remain within the measured
                # radial clearance, rather than demanding an arbitrary roll.
                if attempt>=round(self.args.physics_hz*.10)-1 and axial_error<axial_tolerance and max(m['radial_error_m'],m['entry_radial_error_m'])<.00035 and m['axis_error_rad']<axis_tolerance:
                    break
            else:
                m=insertion_metrics(self.object_pose(obj),self.holes[index],self.normal)
                raise TrialFailed(f'Screw {index+1} insertion stalled at {fraction:.3f}: error={d:.6f}m angle={a:.6f}, {m}')
            if number % max(1,count//9)==0:
                print(f'ASSEMBLY_INSERT screw={index+1} fraction={fraction:.3f} depth={m["depth_m"]:.5f} radial={m["radial_error_m"]:.6f}',flush=True)

    def contact_status(self,index):
        screw=self.object_pose(self.screws[index])
        m=insertion_metrics(screw,self.holes[index],self.normal)
        m.update(keel_hole_metrics(screw,self.object_pose(self.keel),index,self.args.bolt_shoulder_radius or .004))
        # A released clearance-fit pin may settle onto its bore. These limits
        # allow the 0.5 mm physical radial clearance; they do not lock its pose.
        retained=(.020<m['depth_m']<.0265 and max(m['radial_error_m'],m['entry_radial_error_m'])<.00055
            and m['axis_error_rad']<.05 and m['keel_shaft_crosses_plate']
            # This check follows right-hand release. A pin can bear on the
            # bore even while the left hand continues supporting the plate;
            # the pre-release insertion check above still requires a margin.
            and m['keel_hole_radial_error_m']<m['keel_hole_clearance_m']+.00010)
        return dict(screw_index=index+1,retained=bool(retained),**m)

    def verify_supported(self,order,require_retained=True):
        self.physics_check=verify_dynamic_parts(self.stage,self.part_paths)
        d,a=errors(self.object_pose(self.keel),self.keel_target)
        if not self.hold_keel or d>.001 or a>.01:
            raise TrialFailed(f'Left arm did not maintain physical keel support: {d:.6f}m {a:.6f}rad')
        screws=[self.contact_status(index) for index in order]
        if require_retained and not all(m['retained'] for m in screws):
            raise TrialFailed(f'A released screw moved out of its actual hole: {screws}')
        gap=float(np.diff(self.arms[0]['robot'].get_joint_positions()[self.arms[0]['fingers']])[0]*-1)
        if not .6*.038<gap<1.4*.038:raise TrialFailed(f'Left support grasp lost: gap={gap}')
        print(f'ASSEMBLY_CHECK supported_insertion screws={len(screws)} keel_error={d:.6f}m',flush=True)
        return dict(keel_position_error_m=d,keel_angle_error_rad=a,screws=screws,
            left_support_active=True,left_gap_m=gap,independently_mounted=False)

    def observe_release(self,order):
        # Stop following the object before opening the hand.
        # Do not chase or reattach a falling part to make this check pass.
        start=self.object_pose(self.keel)
        self.release_observation=dict(start_pose=start.tolist(),independently_mounted=False)
        self.hold_keel=False
        self.arms[0].pop('posture_seed',None)
        self.left_release_time=self.tick/self.args.physics_hz
        self.release_observation['release_time_s']=self.left_release_time
        self.release_observation['screw_indices']=[i+1 for i in order]
        self.open_at_measured_pose(0,'keel_free_release')
        retreat=self.tcp(0);retreat[:3]+=.12*self.normal
        self.move(0,retreat,'keel_release_retreat',speed=.08)
        self.wait(3.,'gravity_observation')
        actual=self.object_pose(self.keel)
        d,a=errors(actual,self.keel_target)
        screws=[self.contact_status(index) for index in order]
        touching=[pair for pair,tick in self.recent_contacts.items() if tick>self.tick-self.args.physics_hz
            and self.keel_path in pair and any(any(p.startswith(arm['root']+'/') for arm in self.arms) for p in pair)]
        gaps=[float(arm['robot'].get_joint_positions()[arm['fingers']][0]-
            arm['robot'].get_joint_positions()[arm['fingers']][1]) for arm in self.arms]
        mounted=(d<.002 and a<.02 and all(m['retained'] for m in screws) and not touching and min(gaps)>.060)
        self.release_observation.update(final_pose=actual.tolist(),vertical_drop_m=float(start[2]-actual[2]),
            position_error_m=d,angle_error_rad=a,screws=screws,hand_contacts=touching,
            gripper_gaps_m=gaps,independently_mounted=bool(mounted))
        print('ASSEMBLY_GRAVITY '+json.dumps(self.release_observation),flush=True)
        if not mounted:raise TrialFailed('Free release did not maintain keel pose; see release_observation. No fixed joints are used.')
        return self.release_observation

    def settle_on_right_pair(self):
        # Lower only the robot's held-pose command, allowing the actual
        # plate to establish bearing contact before the hand opens.
        # No part is repositioned and no retaining force is applied.
        self.keel_support_target=self.keel_target.copy()
        self.keel_support_target[2]-=self.args.handoff_settle
        self.wait(3.,'keel_bearing_settle')
        actual=self.object_pose(self.keel)
        d,a=errors(actual,self.keel_target)
        if d>.002 or a>.01:
            raise TrialFailed(f'Bearing settle lost plate alignment: {d:.6f}m {a:.6f}rad')
        return dict(commanded_drop_m=self.args.handoff_settle,actual_drop_m=float(self.keel_target[2]-actual[2]),
            position_error_m=d,angle_error_rad=a)

    def park_left(self):
        # Return the empty arm to a reachable raised home pose, leaving
        # room for the right wrist to access the two left-hand holes.
        p,r=self.arms[0]['solver'].compute_forward_kinematics('gripper_base',np.array([0.,-.4,0.,1.,0.,.2,0.]))
        self.move(0,pose(p,Rotation.from_matrix(r)),'left_park',speed=.04)

    def verify_unassisted(self,order):
        self.physics_check=verify_dynamic_parts(self.stage,self.part_paths)
        d,a=errors(self.object_pose(self.keel),self.keel_target)
        screws=[self.contact_status(index) for index in order]
        gaps=[float(arm['robot'].get_joint_positions()[arm['fingers']][0]-
            arm['robot'].get_joint_positions()[arm['fingers']][1]) for arm in self.arms]
        touching=[pair for pair,tick in self.recent_contacts.items() if tick>self.tick-self.args.physics_hz
            and self.keel_path in pair]
        stable=bool(not self.hold_keel and d<.002 and a<.02 and all(m['retained'] for m in screws)
            and not touching and min(gaps)>.060)
        result=dict(keel_position_error_m=d,keel_angle_error_rad=a,screws=screws,
            gripper_gaps_m=gaps,hand_contacts=touching,independently_mounted=stable)
        if not stable:raise TrialFailed(f'Unassisted assembly moved: {result}')
        return result

    def preflight(self):
        # Representative endpoints, including both bottom/top and left/right holes.
        left=keel_grasp(self.keel_start,self.args.left_grip_x,np.deg2rad(self.args.left_tilt_deg))
        relative=compose(inverse(self.keel_start),left)
        targets=[left,compose(self.keel_target,relative)]
        solved=[]
        for i,t in enumerate(targets):solved.append(self.solve(0,t,retry=True).tolist())
        for i,obj in enumerate(self.screws):
            start=self.object_pose(obj)
            world_offset=(0,np.sign(HOLES[i,0])*self.args.screw_grip_offset,self.args.screw_head_height-.007)
            pick=top_grasp(pose(start[:3],Rotation.identity()),world_offset,np.pi/2)
            rel=compose(inverse(start),pick)
            aligned=canonical_screw_target(self.screw_targets[i],rel)
            approach=aligned.copy();approach[:3]+=self.args.approach_distance*self.normal
            for target in (pick,compose(approach,rel),compose(aligned,rel)):
                solved.append(self.solve(1,target,retry=True).tolist())
        print('ASSEMBLY_CHECK preflight endpoints reachable',flush=True)
        return solved

    def save_recording(self,path,success):
        if not self.rows:return
        get=lambda k:np.array([r[k] for r in self.rows])
        np.savez_compressed(path,times=get('time'),tcp_poses=get('tcp'),desired_tcp_poses=get('desired'),
            joint_positions=get('joints'),joint_targets=get('joint_targets'),finger_joint_positions=get('fingers'),gripper_commands=get('opening'),
            keel_poses=get('keel'),screw_poses=get('screws'),stage_ids=get('stage'),stage_names=np.array(self.labels),
            insertion_states=get('inserted'),left_support_active=get('support'),
            keel_linear_velocities=get('keel_velocity'),screw_linear_velocities=get('screw_velocities'),
            active_screw=get('screw'),wall_pose=self.wall,wall_holes=self.holes,keel_target=self.keel_target,
            screw_targets=self.screw_targets,task='dual_keel_friction_insertion_v3',base_frame='World',tcp_frame='gripper_base',
            quaternion_order='wxyz',source='isaacsim-5.1',synthetic=True,success=success,
            screw_head_height_m=self.args.screw_head_height,wall_offset_x_m=self.args.wall_offset_x,
            keel_offset_x_m=self.args.keel_offset_x,grasp_model='physx_contact',
            rack_offset_y_m=self.args.rack_offset_y,
            fastening_model=self.retention_model['model'],assembly_joints=0,gravity_enabled=True,
            independently_mounted=bool(success and self.release_observation and self.release_observation['independently_mounted']),
            anchor_model='simplified_friction_bore' if self.args.bolt_retention=='friction' else 'not_modelled',
            friction_retention_verified=bool(self.retention_checks and all(r['verified'] for r in self.retention_checks)),
            bore_interference_m=self.args.bore_interference,bore_static_friction=self.args.bore_static_friction,
            bore_dynamic_friction=self.args.bore_dynamic_friction,bore_stiffness_N_per_m=self.args.bore_stiffness,
            bore_damping_Ns_per_m=self.args.bore_damping,anchoring_verified=False,
            keel_static_friction=self.args.keel_static_friction,keel_dynamic_friction=self.args.keel_dynamic_friction,
            installation_complete=bool(success and self.args.stage=='full' and self.args.screw_count==4),
            release_after_right_pair=self.args.stage in ('full','right-pair'),
            left_release_time_s=self.left_release_time,
            handoff_settle_m=self.args.handoff_settle,
            bolt_shoulder_radius_m=self.args.bolt_shoulder_radius,
            bolt_shoulder_height_m=.003 if self.args.bolt_shoulder_radius else 0.,
            bolt_shoulder_lead_in_m=np.array([self.args.bolt_shoulder_lead_in]*2+[0.,0.]) if self.args.bolt_shoulder_radius else np.zeros(4),
            finger_stiffness_N_per_m=np.array([600.,self.args.finger_stiffness]),finger_effort_limit_N=10.,
            approach_distance_m=self.args.approach_distance,
            collision_model='compound_keel_open_holes')


def collect(app,args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'settings.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    sim=AssemblySimulation(app,args)
    rng=np.random.default_rng(args.seed)
    reports=[]
    for i in range(args.demos):
        report=dict(episode=i+1,stage=args.stage,success=False,screws=[],
            expected_screws=2 if args.stage=='right-pair' else 1 if args.stage=='insert-one' else args.screw_count if args.stage in ('full','insert-all','release-test') else 0,
            fastening_model=sim.retention_model['model'],anchor_model='simplified_friction_bore' if args.bolt_retention=='friction' else 'not_modelled',independently_mounted=False,
            anchoring_verified=False,installation_complete=False)
        folder=args.output_dir/f'episode_{i+1:03d}';folder.mkdir()
        try:
            variation=rng.uniform(-args.keel_jitter,args.keel_jitter,size=2)
            report['keel_variation_xy_m']=variation.tolist()
            sim.initialize(variation)
            if args.stage=='preflight':
                report['ik_solutions']=sim.preflight()
            elif args.stage=='screw-grasp':
                sim.grab_screw(args.first_screw-1)
            else:
                sim.grab_keel()
                if args.stage!='keel-grasp':
                    sim.align_keel()
                    report['keel_alignment']=sim.alignment_check
                if args.stage in ('insert-one','full','right-pair','insert-all','release-test'):
                    count=report['expected_screws']
                    order=([i-1 for i in screw_order(args.first_screw,count)] if args.first_screw in (3,4) else
                        ([args.first_screw-1]+[i for i in (3,2,0,1) if i!=args.first_screw-1])[:count])
                    report['screw_order']=[n+1 for n in order]
                    for ordinal,n in enumerate(order,1):
                        relative=sim.grab_screw(n)
                        report['screws'].append(sim.insert_screw(n,relative))
                        if args.stage in ('full','right-pair') and ordinal==2:
                            report['supported_check']=sim.verify_supported(order[:2])
                            report['bearing_settle']=sim.settle_on_right_pair()
                            report['right_pair_release']=sim.observe_release(order[:2])
                            sim.park_left()
                    if args.stage in ('full','right-pair'):
                        sim.wait(args.retention_seconds,'unassisted_final_hold')
                        report['mounted_check']=sim.verify_unassisted(order)
                        report['independently_mounted']=True
                        report['installation_complete']=args.stage=='full' and count==4
                    else:
                        sim.wait(2.,'supported_hold')
                        report['supported_check']=sim.verify_supported(order,require_retained=args.stage!='release-test')
                    if args.stage=='release-test':
                        report['release_observation']=sim.observe_release(order)
                        report['independently_mounted']=report['release_observation']['independently_mounted']
            report['success']=True
        except (Exception,KeyboardInterrupt) as exc:
            report['error']=str(exc) or type(exc).__name__
            report['interrupted']=isinstance(exc,KeyboardInterrupt)
            print(f'ASSEMBLY_FAILED episode={i+1}: {exc}',flush=True)
        report['last_stage']=sim.label
        report['grasp_checks']=sim.grasp_checks
        report['insertion_events']=sim.insertion_events
        report['physics_check']=sim.physics_check
        report['material_check']=sim.material_check
        report['bolt_geometry']=dict(shaft_radius_m=.004,shaft_length_m=.028,head_height_m=args.screw_head_height,
            shoulder_radius_m=args.bolt_shoulder_radius,shoulder_height_m=.003 if args.bolt_shoulder_radius else 0.,
            shoulder_lead_in_m=[args.bolt_shoulder_lead_in]*2+[0.,0.] if args.bolt_shoulder_radius else [0.]*4)
        report['retention_model']={k:v for k,v in sim.retention_model.items() if k!='collider_paths'}
        report['retention_checks']=sim.retention_checks
        report['friction_retention_verified']=bool(sim.retention_checks and all(r['verified'] for r in sim.retention_checks))
        if sim.release_observation is not None:report['release_observation']=sim.release_observation
        report['samples']=len(sim.rows)
        sim.save_recording(folder/'trajectory.npz',report['success'])
        write_json(folder/'report.json',report);reports.append(report)
        write_json(args.output_dir/'summary.json',dict(task='dual_keel_friction_insertion_v3',requested=args.demos,
            successful=sum(r['success'] for r in reports),episodes=reports))
        if report.get('interrupted') or not app.is_running():break
    print(f'ASSEMBLY_DONE stage={args.stage} successful={sum(r["success"] for r in reports)}/{args.demos} '
        f'installation_complete={sum(r["installation_complete"] for r in reports)} anchoring_verified=False output={args.output_dir}',flush=True)
    if args.keep_open and not args.headless:
        sim.record=False
        if not all(r['success'] for r in reports):
            sim.world.pause()
        while app.is_running():
            if sim.world.is_playing():sim.step()
            else:app.update()
    return 0 if len(reports)==args.demos and all(r['success'] for r in reports) else 1
