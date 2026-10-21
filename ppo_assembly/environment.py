"""Isaac Lab vectorized single-bolt insertion with contact-only retention."""
import csv
import json
from collections import deque
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController,DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.utils import configclass
from ppo_assembly.paths import ASSETS
from ppo_assembly.task_math import apply,qconj,qmul,rotvec,axis_quat,bolt_metrics,centered,retained
from ppo_assembly.position_force import ACTION_DIM,OBSERVATION_DIM,insertion_filter_paths,axial_velocity,force_costs


@configclass
class BoltInsertionCfg(DirectRLEnvCfg):
    decimation=12
    episode_length_s=35.
    action_space=ACTION_DIM
    observation_space=OBSERVATION_DIM
    state_space=0
    is_finite_horizon=True
    ui_window_class_type=None
    sim=sim_utils.SimulationCfg(dt=1/120,render_interval=12,
        gravity=(0.,0.,-9.81),physx=sim_utils.PhysxCfg(solver_type=1,
            max_position_iteration_count=64,max_velocity_iteration_count=8,
            gpu_max_rigid_contact_count=2**22,gpu_max_rigid_patch_count=2**20))
    scene=InteractiveSceneCfg(num_envs=16,env_spacing=3.,replicate_physics=True,clone_in_fabric=False)
    difficulty=0
    automatic_curriculum=False
    evaluation=False
    retention_seconds=3.
    curriculum_min_episodes=50
    curriculum_success_rate=.75
    axial_action_m=.0008
    lateral_action_m=.0004
    tilt_action_rad=.003
    joint_speed_limit=.8
    force_reference_min_N=2.
    force_reference_max_N=8.
    expert_force_reference_N=7.5
    force_filter_time_s=.04
    axial_admittance_m_per_N_s=.004
    contact_threshold_N=.05
    lateral_yield_N=2.
    axial_soft_limit_N=8.
    hard_contact_limit_N=12.
    hard_lateral_limit_N=5.
    force_scale_N=5.
    lateral_force_scale_N=2.
    force_tracking_weight=.03
    lateral_force_weight=.04
    overload_force_weight=.08
    force_variation_weight=.005
    time_cost=.005
    action_variation_weight=.0002


class BoltInsertionEnv(DirectRLEnv):
    cfg:BoltInsertionCfg

    def __init__(self,cfg,render_mode=None,**kwargs):
        self.fixture=json.loads((ASSETS/'fixture.json').read_text())
        super().__init__(cfg,render_mode,**kwargs)
        self.arm_ids=[];self.finger_ids=[];self.ee_ids=[];self.controllers=[]
        for robot in (self.left,self.right):
            self.arm_ids.append(robot.find_joints([f'joint{i}' for i in range(1,8)],preserve_order=True)[0])
            self.finger_ids.append(robot.find_joints(['gripper_joint1','gripper_joint2'],preserve_order=True)[0])
            self.ee_ids.append(robot.find_bodies('gripper_base')[0][0])
            self.controllers.append(DifferentialIKController(DifferentialIKControllerCfg(
                command_type='pose',use_relative_mode=False,ik_method='dls',ik_params={'lambda_val':.025}),
                self.num_envs,self.device))
        t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
        n=self.num_envs
        self.basis=t(self.fixture['basis']);self.normal=self.basis[:,0].expand(n,3)
        self.horizontal=self.basis[:,1].expand(n,3);self.up=self.basis[:,2].expand(n,3)
        self.goal=t(self.fixture['bolt_goal']).expand(n,7).clone();self.goal[:,:3]+=self.scene.env_origins
        self.keel_goal=t(self.fixture['keel_target']).expand(n,7).clone();self.keel_goal[:,:3]+=self.scene.env_origins
        self.hole=t(self.fixture['wall_holes'][3]).expand(n,3)+self.scene.env_origins
        self.expected_grasp=t(self.fixture['tcp_to_bolt'][:3]).expand(n,3)
        self.left_initial=t(self.fixture['joint_positions'][0])
        self.left_pose=t(self.fixture['desired_tcp_poses'][0]).expand(n,7).clone()
        self.left_pose[:,:3]+=self.scene.env_origins
        self.hand_targets=[self.left_pose.clone(),torch.zeros(n,7,device=self.device)]
        with np.load(ASSETS/'reset_states.npz') as s:
            self.reset_bolts=t(s['bolt_poses']);self.reset_tcp=t(s['tcp_poses']);self.reset_q=t(s['joint_positions'])
            self.pools=[torch.as_tensor(np.flatnonzero((s['levels']==level)&(s['splits']==int(cfg.evaluation))),
                dtype=torch.long,device=self.device) for level in range(3)]
        self.phase=torch.zeros(n,dtype=torch.long,device=self.device) # settle, policy, open, retreat, free observation
        self.phase_ticks=torch.zeros_like(self.phase)
        self.control_phase=self.phase.clone()
        self.center_ticks=torch.zeros_like(self.phase)
        self.previous_actions=torch.zeros(n,ACTION_DIM,device=self.device)
        self.actions=torch.zeros_like(self.previous_actions)
        self.episode_returns=torch.zeros(n,device=self.device)
        self.level=int(cfg.difficulty);self.recent_success=deque(maxlen=50)
        self.episode_levels=torch.full((n,),self.level,dtype=torch.long,device=self.device)
        self.completed_episodes=0;self.records=[];self.csv_writer=None;self.csv_file=None
        self.position_min=torch.zeros(n,3,device=self.device);self.position_max=torch.zeros_like(self.position_min)
        self.verify_start=torch.zeros_like(self.position_min)
        self.retention_extent=torch.zeros(n,device=self.device)
        self.done_success=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.done_failure=torch.zeros_like(self.done_success)
        self.fail_code=torch.zeros(n,dtype=torch.long,device=self.device)
        self.slew=torch.zeros(n,device=self.device)
        self.phi=torch.zeros(n,device=self.device);self.old_phi=self.phi.clone()
        self.force_raw=torch.zeros(n,3,device=self.device);self.force_filtered=self.force_raw.clone()
        self.old_force=self.force_raw.clone();self.contact_load=torch.zeros(n,device=self.device)
        self.force_reference=torch.full((n,),cfg.expert_force_reference_N,device=self.device)
        self.step_force_peak=torch.zeros(n,device=self.device);self.step_lateral_peak=self.step_force_peak.clone()
        self.episode_force_peak=self.step_force_peak.clone();self.episode_lateral_peak=self.step_force_peak.clone()
        self.contacted=torch.zeros(n,dtype=torch.bool,device=self.device)
        self.force_normal_world=self.force_raw.clone();self.force_friction_world=self.force_raw.clone()
        self.goal_depth=bolt_metrics(self.goal,self.keel_goal,self.hole,self.normal)['depth']
        self.outcome_ready=False
        self.asset_check=self._check_physics()
        print('PPO_ENV '+json.dumps(dict(num_envs=n,observations=OBSERVATION_DIM,actions=ACTION_DIM,
            force_source='filtered_wall_and_keel_normal_plus_friction',
            reset_prepositioned=True,includes_pickup=False,installation_complete=False,**self.asset_check)),flush=True)

    def _setup_scene(self):
        from isaacsim.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(str(ASSETS/'cell.usdc'),'/World/envs/env_0/Cell')
        prefix='/World/envs/env_.*/Cell'
        def robot_cfg(root,y):
            return ArticulationCfg(prim_path=prefix+root.removeprefix('/World'),spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.,y,1.)),
                actuators={
                    'arm':ImplicitActuatorCfg(joint_names_expr=['joint[1-7]'],stiffness=20000.,damping=600.,
                        effort_limit_sim=100.,velocity_limit_sim=1.),
                    'fingers':ImplicitActuatorCfg(joint_names_expr=['gripper_joint.*'],stiffness=600.,damping=10.,
                        effort_limit_sim=10.,velocity_limit_sim=.05)})
        self.left=Articulation(robot_cfg(self.fixture['left_robot_root'],-.375))
        self.right=Articulation(robot_cfg(self.fixture['right_robot_root'],.375))
        env=prefix+'/UserScene/Environment'
        self.keel=RigidObject(RigidObjectCfg(prim_path=env+'/Geometry',spawn=None))
        self.bolt=RigidObject(RigidObjectCfg(prim_path=env+'/Screw04',spawn=None))
        self.scene.clone_environments(copy_from_source=False)
        if self.device=='cpu':self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations['left']=self.left;self.scene.articulations['right']=self.right
        self.scene.rigid_objects['keel']=self.keel;self.scene.rigid_objects['bolt']=self.bolt
        rightroot=prefix+self.fixture['right_robot_root'].removeprefix('/World')
        leftroot=prefix+self.fixture['left_robot_root'].removeprefix('/World')
        self.arm_contacts=ContactSensor(ContactSensorCfg(prim_path=rightroot+'/(link[1-7]|gripper_base|gripper_flange)',
            history_length=1,force_threshold=.1))
        self.bolt_contacts=ContactSensor(ContactSensorCfg(prim_path=env+'/Screw04',history_length=1,
            filter_prim_paths_expr=[r+f'/gripper_link{i}' for r in (leftroot,rightroot) for i in (1,2)]))
        self.scene.sensors['arm_contacts']=self.arm_contacts;self.scene.sensors['bolt_contacts']=self.bolt_contacts
        self.insertion_contacts=ContactSensor(ContactSensorCfg(prim_path=env+'/Screw04',update_period=0.,
            history_length=1,track_friction_forces=True,max_contact_data_count_per_prim=1024,
            filter_prim_paths_expr=insertion_filter_paths(env)))
        self.scene.sensors['insertion_contacts']=self.insertion_contacts
        light=sim_utils.DomeLightCfg(intensity=900.)
        light.func('/World/Light',light)

    def _check_physics(self):
        from assembly_collection.physics import verify_dynamic_parts
        from isaacsim.core.utils.stage import get_current_stage
        stage=get_current_stage()
        paths=[f'/World/envs/env_{i}/Cell/UserScene/Environment/{name}'
            for i in range(self.num_envs) for name in ('Geometry','Screw04')]
        return verify_dynamic_parts(stage,paths)

    def _ee(self,index):
        r=(self.left,self.right)[index];b=self.ee_ids[index]
        return r.data.body_pos_w[:,b],r.data.body_quat_w[:,b]

    def _state(self):
        bolt=torch.cat((self.bolt.data.root_pos_w,self.bolt.data.root_quat_w),dim=-1)
        keel=torch.cat((self.keel.data.root_pos_w,self.keel.data.root_quat_w),dim=-1)
        return bolt,keel

    def _measure(self):
        bolt,keel=self._state();m=bolt_metrics(bolt,keel,self.hole,self.normal)
        p,q=self._ee(1)
        relative=apply(qconj(q),bolt[:,:3]-p)
        m['slip']=torch.linalg.vector_norm(relative-self.expected_grasp,dim=-1)
        f=self.right.data.joint_pos[:,self.finger_ids[1]]
        m['gap']=f[:,0]-f[:,1]
        m['keel_pos_error']=torch.linalg.vector_norm(keel[:,:3]-self.keel_goal[:,:3],dim=-1)
        m['keel_angle_error']=torch.linalg.vector_norm(rotvec(qmul(keel[:,3:7],qconj(self.keel_goal[:,3:7]))),dim=-1)
        m['hand_force']=torch.linalg.vector_norm(self.bolt_contacts.data.force_matrix_w,dim=-1).sum(dim=(1,2))
        m['arm_force']=torch.linalg.vector_norm(self.arm_contacts.data.net_forces_w,dim=-1).amax(dim=1)
        m['axial_force']=self.force_raw[:,0].clamp(min=0.)
        m['lateral_force']=torch.linalg.vector_norm(self.force_raw[:,1:],dim=-1)
        m['interface_force']=torch.linalg.vector_norm(self.force_raw,dim=-1)
        m['contact_load']=self.contact_load
        return m

    def _potential(self,m,goal_depth=None):
        goal=self.goal_depth if goal_depth is None else goal_depth
        return 4*torch.tanh(m['depth']/.03)-2*torch.tanh(torch.maximum(m['radial'],m['entry'])/.001)-.5*torch.tanh(m['axis_error']/.05)-torch.tanh((m['depth']-goal).abs()/.03)

    def _update_force(self,dt):
        data=self.insertion_contacts.data
        normal=data.force_matrix_w;friction=data.friction_forces_w
        self.force_normal_world=normal.sum((1,2));self.force_friction_world=friction.sum((1,2))
        raw=(self.force_normal_world+self.force_friction_world)@self.basis
        if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite insertion contact force')
        self.force_raw=raw
        self.contact_load=torch.linalg.vector_norm(normal+friction,dim=-1).sum((1,2))
        self.contacted=self.contact_load>self.cfg.contact_threshold_N
        if dt:
            alpha=1.-np.exp(-dt/self.cfg.force_filter_time_s)
            self.force_filtered+=alpha*(raw-self.force_filtered)
        magnitude=torch.linalg.vector_norm(raw,dim=-1);side=torch.linalg.vector_norm(raw[:,1:],dim=-1)
        self.step_force_peak=torch.maximum(self.step_force_peak,magnitude)
        self.step_lateral_peak=torch.maximum(self.step_lateral_peak,side)
        self.episode_force_peak=torch.maximum(self.episode_force_peak,magnitude)
        self.episode_lateral_peak=torch.maximum(self.episode_lateral_peak,side)

    def _pre_physics_step(self,actions):
        self.old_phi=self.phi.clone()
        self.control_phase=self.phase.clone()
        self.old_force=self.force_filtered.clone()
        self.step_force_peak.zero_();self.step_lateral_peak.zero_()
        self.actions=actions.clamp(-1,1).clone()
        self.slew=(self.actions-self.previous_actions).square().sum(-1)
        active=self.phase==1
        self.force_reference=self.cfg.force_reference_min_N+(self.actions[:,5]+1.)*.5*(self.cfg.force_reference_max_N-self.cfg.force_reference_min_N)
        delta=self.horizontal*self.actions[:,1:2]*self.cfg.lateral_action_m+self.up*self.actions[:,2:3]*self.cfg.lateral_action_m
        angle=(self.horizontal*self.actions[:,3:4]+self.up*self.actions[:,4:5])*self.cfg.tilt_action_rad
        # Rotate the commanded grasp about the virtual bolt centre, not the
        # wrist 168 mm behind it. This is a Cartesian action mapping; it uses
        # the stored grasp geometry and never writes the actual object pose.
        old_rotation=self.hand_targets[1][active,3:7].clone()
        new_rotation=qmul(axis_quat(angle[active]),old_rotation)
        offset=self.expected_grasp[active]
        self.hand_targets[1][active,:3]+=delta[active]+apply(old_rotation,offset)-apply(new_rotation,offset)
        self.hand_targets[1][active,3:7]=new_rotation
        # Bound the command/hand separation to avoid accumulating a large
        # drive target behind a blocked bolt. This does not move the object.
        p,q=self._ee(1);lag=self.hand_targets[1][:,:3]-p
        length=torch.linalg.vector_norm(lag,dim=-1,keepdim=True)
        self.hand_targets[1][:,:3]=p+lag*torch.minimum(torch.ones_like(length),.004/length.clamp(min=1e-8))
        retreat=self.phase==3
        self.hand_targets[1][retreat,:3]+=.03*self.step_dt*self.normal[retreat]
        self.previous_actions=self.actions.clone()
        # The left hand maintains physical support; the learned right hand
        # gets no automatic bolt-pose correction during the policy phase.
        kp=self.keel.data.root_pos_w;kq=self.keel.data.root_quat_w
        correction=rotvec(qmul(self.keel_goal[:,3:7],qconj(kq)))
        self.hand_targets[0][:,:3]+=.2*(self.keel_goal[:,:3]-kp)
        self.hand_targets[0][:,3:7]=qmul(axis_quat(.2*correction),self.hand_targets[0][:,3:7])

    def _apply_action(self):
        # Read actual contact force at physics rate, then integrate a compliant
        # axial velocity. The target-force action never writes a force to the bolt.
        self._update_force(self.physics_dt)
        maximum=self.cfg.axial_action_m/self.step_dt
        feedback=self.force_filtered.clone()
        feedback[:,0]=torch.maximum(feedback[:,0],self.force_raw[:,0])
        velocity=axial_velocity(self.actions[:,0]*maximum,feedback,self.force_reference,self.contacted,
            maximum,self.cfg.axial_admittance_m_per_N_s,self.cfg.lateral_yield_N)
        active=self.phase==1
        self.hand_targets[1][active,:3]-=self.normal[active]*velocity[active,None]*self.physics_dt
        p,_=self._ee(1);lag=self.hand_targets[1][:,:3]-p
        length=torch.linalg.vector_norm(lag,dim=-1,keepdim=True)
        self.hand_targets[1][:,:3]=p+lag*torch.minimum(torch.ones_like(length),.004/length.clamp(min=1e-8))
        for index,r in enumerate((self.left,self.right)):
            p,q=self._ee(index);ids=self.arm_ids[index]
            jac=r.root_physx_view.get_jacobians()[:,self.ee_ids[index]-int(r.is_fixed_base),:,:][:,:,ids]
            controller=self.controllers[index];controller.set_command(self.hand_targets[index])
            actual=r.data.joint_pos[:,ids]
            wanted=controller.compute(p,q,jac,actual)
            target=actual+(wanted-actual).clamp(-self.cfg.joint_speed_limit*self.physics_dt,self.cfg.joint_speed_limit*self.physics_dt)
            bounds=r.data.soft_joint_pos_limits[:,ids]
            target=torch.maximum(torch.minimum(target,bounds[:,:,1]-.001),bounds[:,:,0]+.001)
            r.set_joint_position_target(target,joint_ids=ids)
            gap=torch.zeros(self.num_envs,2,device=self.device)
            if index==1:
                released=self.phase>=2;gap[released,0]=.035;gap[released,1]=-.035
            r.set_joint_position_target(gap,joint_ids=self.finger_ids[index])
            r.set_joint_effort_target(r.root_physx_view.get_gravity_compensation_forces())

    def _get_dones(self):
        self._update_force(0.)
        m=self._measure();self.m=m;self.phase_ticks+=1
        warm=(self.phase==0)&(self.phase_ticks>=round(.5/self.step_dt))
        self.phase[warm]=1;self.phase_ticks[warm]=0
        good=centered(m)&(self.phase==1)
        self.center_ticks=torch.where(good,self.center_ticks+1,torch.zeros_like(self.center_ticks))
        release=(self.phase==1)&(self.center_ticks>=5)
        p,q=self._ee(1)
        self.hand_targets[1][release]=torch.cat((p,q),dim=-1)[release]
        self.phase[release]=2;self.phase_ticks[release]=0
        retreat=(self.phase==2)&(self.phase_ticks>=round(1.5/self.step_dt))
        self.phase[retreat]=3;self.phase_ticks[retreat]=0
        free=(self.phase==3)&(self.phase_ticks>=round(2./self.step_dt))
        self.phase[free]=4;self.phase_ticks[free]=0
        bp=self.bolt.data.root_pos_w
        self.verify_start[free]=bp[free];self.position_min[free]=bp[free];self.position_max[free]=bp[free]
        observing=self.phase==4
        self.position_min[observing]=torch.minimum(self.position_min[observing],bp[observing])
        self.position_max[observing]=torch.maximum(self.position_max[observing],bp[observing])
        end=observing&(self.phase_ticks>=round(self.cfg.retention_seconds/self.step_dt))
        extent=torch.linalg.vector_norm(self.position_max-self.position_min,dim=-1)
        self.retention_extent=extent
        distance=torch.linalg.vector_norm(p-bp,dim=-1)
        self.done_success=end&retained(m)&(m['gap']>.060)&(m['hand_force']<.05)&(distance>.06)&(
            extent<.00025)&(m['keel_pos_error']<.001)&(m['keel_angle_error']<.01)
        error=(bp-self.goal[:,:3])@self.basis
        dropped=((self.phase<2)&(m['slip']>.006))|(bp[:,2]<.95)
        bounds=(error[:,1:].abs().amax(-1)>.010)|(error[:,0]>.085)|(error[:,0]<-.006)
        support=(m['keel_pos_error']>.005)|(m['keel_angle_error']>.08)
        collision=m['arm_force']>20.
        overload=(self.control_phase<=1)&((self.step_force_peak>self.cfg.hard_contact_limit_N)|(
            self.step_lateral_peak>self.cfg.hard_lateral_limit_N))
        self.done_failure=dropped|bounds|support|collision|overload|(end&~self.done_success)
        self.fail_code=torch.where(dropped,1,torch.where(bounds,2,torch.where(support,3,torch.where(collision,4,torch.where(overload,6,torch.where(end&~self.done_success,5,0))))))
        self.phi=self._potential(m)
        timeout=self.episode_length_buf>=self.max_episode_length-1
        self.outcome_ready=True
        return self.done_success|self.done_failure,timeout

    def _get_rewards(self):
        active=(self.control_phase==1).float()
        costs=force_costs(self.force_filtered,self.old_force,self.force_reference,self.contacted,
            (self.actions[:,0]>0.)&(self.m['depth']<self.goal_depth-.0005),self.cfg.axial_soft_limit_N,
            self.cfg.force_scale_N,self.cfg.lateral_force_scale_N)
        reward=.99*self.phi-self.old_phi-self.cfg.time_cost
        self.force_penalty=active*(self.cfg.force_tracking_weight*costs['tracking']+
            self.cfg.lateral_force_weight*costs['lateral']+self.cfg.overload_force_weight*costs['overload']+
            self.cfg.force_variation_weight*costs['variation'])
        reward-=self.force_penalty+active*self.cfg.action_variation_weight*self.slew
        reward+=100*self.done_success.float()-30*self.done_failure.float()
        self.episode_returns+=reward
        self.extras['log']={'Task/depth_mm':self.m['depth'].mean()*1000,
            'Task/radial_error_mm':self.m['entry'].mean()*1000,'Task/curriculum_level':float(self.level),
            'Task/recent_success_rate':sum(self.recent_success)/max(1,len(self.recent_success))}
        self.extras['log'].update({'Force/axial_N':self.m['axial_force'].mean(),
            'Force/lateral_N':self.m['lateral_force'].mean(),'Force/reference_N':self.force_reference.mean(),
            'Force/step_peak_N':self.step_force_peak.mean(),'Reward/force_cost':self.force_penalty.mean()})
        return reward

    def _get_observations(self):
        bolt,keel=self._state();m=self._measure();p,q=self._ee(1)
        ids=self.arm_ids[1];bounds=self.right.data.soft_joint_pos_limits[:,ids]
        qn=2*(self.right.data.joint_pos[:,ids]-bounds[:,:,0])/(bounds[:,:,1]-bounds[:,:,0])-1
        relative=apply(qconj(q),bolt[:,:3]-p)
        error=(self.goal[:,:3]-bolt[:,:3])@self.basis
        axis=m['axis']@self.basis
        obs=torch.cat((qn,self.right.data.joint_vel[:,ids]/2,
            error/torch.tensor([.05,.005,.005],device=self.device),axis[:,1:]/.1,
            self.bolt.data.root_lin_vel_w@self.basis/.1,m['gap'][:,None]/.018,
            (relative-self.expected_grasp)/.005,(keel[:,:3]-self.keel_goal[:,:3])/.002,
            rotvec(qmul(keel[:,3:7],qconj(self.keel_goal[:,3:7])))/.05,
            self.previous_actions,torch.nn.functional.one_hot(self.phase,5).float(),
            (1-self.episode_length_buf.float()/self.max_episode_length)[:,None],
            (self.hand_targets[1][:,:3]-p)/.01,
            rotvec(qmul(self.hand_targets[1][:,3:7],qconj(q)))/.1,
            self.force_raw/self.cfg.force_scale_N,self.force_filtered/self.cfg.force_scale_N,
            (self.force_reference/self.cfg.force_reference_max_N)[:,None],self.contacted[:,None].float()),dim=-1)
        if obs.shape[1]!=OBSERVATION_DIM or not torch.isfinite(obs).all():raise RuntimeError('Invalid PPO observation')
        return {'policy':obs.clamp(-10,10)}

    def _reset_idx(self,env_ids):
        if self.outcome_ready:
            self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)
        count=len(env_ids);pool=self.pools[self.level]
        selected=pool[torch.randint(len(pool),(count,),device=self.device)]
        origins=self.scene.env_origins[env_ids]
        for index,r in enumerate((self.left,self.right)):
            q=r.data.default_joint_pos[env_ids].clone();v=torch.zeros_like(q)
            q[:,self.arm_ids[index]]=self.left_initial if index==0 else self.reset_q[selected]
            fingers=torch.tensor(self.fixture['finger_joint_positions'][index],device=self.device)
            q[:,self.finger_ids[index]]=fingers
            r.write_joint_state_to_sim(q,v,env_ids=env_ids)
            r.set_joint_position_target(q,env_ids=env_ids)
        for obj,values in ((self.keel,torch.as_tensor(self.fixture['keel_poses'],device=self.device).expand(count,7)),
                           (self.bolt,self.reset_bolts[selected])):
            state=obj.data.default_root_state[env_ids].clone();state[:,:7]=values
            state[:,:3]+=origins;state[:,7:]=0
            obj.write_root_state_to_sim(state,env_ids=env_ids)
        self.hand_targets[0][env_ids]=self.left_pose[env_ids]
        self.hand_targets[1][env_ids]=self.reset_tcp[selected]
        self.hand_targets[1][env_ids,:3]+=origins
        self.phase[env_ids]=0;self.phase_ticks[env_ids]=0;self.center_ticks[env_ids]=0
        self.episode_levels[env_ids]=self.level
        self.previous_actions[env_ids]=0;self.actions[env_ids]=0;self.episode_returns[env_ids]=0
        bp=self.reset_bolts[selected].clone();bp[:,:3]+=origins
        kp=torch.as_tensor(self.fixture['keel_poses'],device=self.device).expand(count,7).clone();kp[:,:3]+=origins
        self.phi[env_ids]=self._potential(bolt_metrics(bp,kp,self.hole[env_ids],self.normal[env_ids]),self.goal_depth[env_ids])
        self.old_phi[env_ids]=self.phi[env_ids]
        self.done_success[env_ids]=False;self.done_failure[env_ids]=False
        self.position_min[env_ids]=0;self.position_max[env_ids]=0
        for buffer in (self.force_raw,self.force_filtered,self.old_force,self.contact_load,self.step_force_peak,
                       self.step_lateral_peak,self.episode_force_peak,self.episode_lateral_peak):buffer[env_ids]=0
        self.contacted[env_ids]=False;self.force_reference[env_ids]=self.cfg.expert_force_reference_N

    def _record_outcomes(self,env_ids):
        reasons=['','grasp_lost','out_of_bounds','keel_support_lost','arm_collision','free_retention_failed','insertion_overload']
        for i in env_ids.tolist():
            success=bool(self.done_success[i]);self.completed_episodes+=1
            if int(self.episode_levels[i])==self.level:self.recent_success.append(success)
            record=dict(episode=self.completed_episodes,env=i,difficulty=int(self.episode_levels[i]),success=success,
                reason='success' if success else reasons[int(self.fail_code[i])] or 'timeout',
                return_value=float(self.episode_returns[i]),steps=int(self.episode_length_buf[i]),
                sim_seconds=float(self.episode_length_buf[i])*self.step_dt,
                depth_mm=float(self.m['depth'][i])*1000,radial_mm=float(self.m['entry'][i])*1000,
                axis_rad=float(self.m['axis_error'][i]),gap_mm=float(self.m['gap'][i])*1000,
                hand_force_N=float(self.m['hand_force'][i]),reset_prepositioned=True,installation_complete=False)
            record.update(free_observation_s=float(self.phase_ticks[i])*self.step_dt if int(self.phase[i])==4 else 0.,
                free_motion_mm=float(self.retention_extent[i])*1000 if int(self.phase[i])==4 else 0.,
                keel_position_error_mm=float(self.m['keel_pos_error'][i])*1000)
            record.update(axial_force_N=float(self.m['axial_force'][i]),lateral_force_N=float(self.m['lateral_force'][i]),
                peak_interface_force_N=float(self.episode_force_peak[i]),peak_lateral_force_N=float(self.episode_lateral_peak[i]))
            self.records.append(record)
            if self.csv_writer:self.csv_writer.writerow(record);self.csv_file.flush()
        if self.cfg.automatic_curriculum and self.level<2 and len(self.recent_success)>=self.cfg.curriculum_min_episodes:
            if sum(self.recent_success)/len(self.recent_success)>=self.cfg.curriculum_success_rate:
                self.level+=1;self.recent_success.clear();print(f'PPO_CURRICULUM level={self.level}',flush=True)
        self.extras['log']['Task/recent_success_rate']=sum(self.recent_success)/max(1,len(self.recent_success))

    def attach_csv(self,path):
        self.csv_file=open(path,'w',newline='')
        keys=['episode','env','difficulty','success','reason','return_value','steps','sim_seconds','depth_mm',
            'radial_mm','axis_rad','gap_mm','hand_force_N','reset_prepositioned','installation_complete',
            'free_observation_s','free_motion_mm','keel_position_error_mm','axial_force_N','lateral_force_N',
            'peak_interface_force_N','peak_lateral_force_N']
        self.csv_writer=csv.DictWriter(self.csv_file,fieldnames=keys);self.csv_writer.writeheader();self.csv_file.flush()

    def expert_action(self):
        m=self._measure();bolt,_=self._state()
        delta=(self.goal[:,:3]-bolt[:,:3])@self.basis
        axis=m['axis'];correction=torch.cross(axis,self.normal,dim=-1)
        a=torch.zeros(self.num_envs,ACTION_DIM,device=self.device)
        a[:,5]=2*(self.cfg.expert_force_reference_N-self.cfg.force_reference_min_N)/(
            self.cfg.force_reference_max_N-self.cfg.force_reference_min_N)-1.
        aligned=(torch.maximum(m['entry'],m['radial'])<.00018)&(m['axis_error']<.006)
        a[:,0]=torch.where(aligned,(-delta[:,0]/self.cfg.axial_action_m*.4).clamp(-.6,.6),torch.zeros_like(delta[:,0]))
        a[:,1]=delta[:,1]/self.cfg.lateral_action_m*.4;a[:,2]=delta[:,2]/self.cfg.lateral_action_m*.4
        a[:,3]=(correction*self.horizontal).sum(-1)/self.cfg.tilt_action_rad*.4
        a[:,4]=(correction*self.up).sum(-1)/self.cfg.tilt_action_rad*.4
        return a.clamp(-1,1)

    def close(self):
        if self.csv_file:self.csv_file.close()
        super().close()
