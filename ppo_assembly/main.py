"""Isaac Lab + RSL-RL PPO training, physical checks and policy replay."""
import argparse
import hashlib
import json
import os
import signal
import shutil
from datetime import datetime
from pathlib import Path
from ppo_assembly.paths import ROOT,ASSETS,local_path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode',choices=['train','play','check'])
parser.add_argument('--num-envs',type=int,default=None)
parser.add_argument('--iterations',type=int,default=2000,help='Additional PPO updates, also when resuming')
parser.add_argument('--steps-per-env',type=int,default=32)
parser.add_argument('--episodes',type=int,default=10)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--difficulty',choices=['easy','medium','hard','auto'],default='easy')
parser.add_argument('--run-dir',default=None)
parser.add_argument('--checkpoint',default=None,help='PPO checkpoint inside ppo_assembly/')
parser.add_argument('--controller',choices=['expert','random','zero'],default='expert',help='check mode only')
parser.add_argument('--max-steps',type=int,default=2000,help='check/play rollout limit')
parser.add_argument('--retention-seconds',type=float,default=None)
parser.add_argument('--save-interval',type=int,default=50)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
cli=parser.parse_args()
if not cli.experience:cli.experience=str(Path(os.environ['ISAAC_PATH'])/'apps/isaacsim.exp.base.python.kit')
if cli.num_envs is not None and cli.num_envs<1:parser.error('num-envs must be positive')
if min(cli.iterations,cli.steps_per_env,cli.episodes,cli.max_steps,cli.save_interval)<1:parser.error('Counts must be positive')
if cli.retention_seconds is not None and cli.retention_seconds<3:parser.error('retention-seconds must be >=3')
if not all((ASSETS/p).is_file() for p in ('cell.usdc','fixture.json','reset_states.npz')):
    parser.error('First run: ./ppo_assembly/run.sh prepare --headless')
if cli.mode=='play' and not cli.checkpoint:parser.error('play requires --checkpoint')
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
out=local_path(cli.run_dir) if cli.run_dir else ROOT/('runs' if cli.mode=='train' else 'evaluations')/(cli.mode+'_'+stamp)
if out.exists() and any(out.iterdir()):parser.error('Use a new/empty run-dir, including when resuming into a new run')
checkpoint=local_path(cli.checkpoint) if cli.checkpoint else None
if checkpoint and not checkpoint.is_file():parser.error('Checkpoint does not exist')
from ppo_assembly.position_force import TASK_ID,ACTION_DIM,OBSERVATION_DIM
if checkpoint:
    import torch
    information=torch.load(checkpoint,map_location='cpu',weights_only=False).get('infos',{})
    if information.get('task')!=TASK_ID:
        parser.error('Checkpoint belongs to the old position-only task. Position/force v2 has 6 actions and 58 observations; start a new training run. Old models are preserved.')
out.mkdir(parents=True,exist_ok=True)
launcher=AppLauncher(dict(vars(cli),create_new_stage=True));app=launcher.app
signal.signal(signal.SIGINT,signal.default_int_handler)


def agent_config(device):
    actor=dict(class_name='MLPModel',hidden_dims=[128,128],activation='elu',obs_normalization=False,
        distribution_cfg=dict(class_name='GaussianDistribution',init_std=.4,std_type='log'))
    critic=dict(class_name='MLPModel',hidden_dims=[128,128],activation='elu',obs_normalization=False,distribution_cfg=None)
    algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
        entropy_coef=.005,num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,
        schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.,
        normalize_advantage_per_mini_batch=False,rnd_cfg=None,symmetry_cfg=None)
    return dict(seed=cli.seed,device=device,num_steps_per_env=cli.steps_per_env,
        save_interval=cli.save_interval,logger='tensorboard',check_for_nan=True,
        obs_groups={'actor':['policy'],'critic':['policy']},actor=actor,critic=critic,algorithm=algorithm)


def run():
    import importlib.metadata
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from ppo_assembly.environment import BoltInsertionCfg,BoltInsertionEnv
    torch.manual_seed(cli.seed);np.random.seed(cli.seed)
    cfg=BoltInsertionCfg();cfg.seed=cli.seed;cfg.sim.device=cli.device
    cfg.scene.num_envs=cli.num_envs or (16 if cli.mode=='train' else 1)
    cfg.difficulty={'easy':0,'medium':1,'hard':2,'auto':0}[cli.difficulty]
    cfg.automatic_curriculum=cli.difficulty=='auto' and cli.mode=='train'
    cfg.evaluation=cli.mode!='train'
    cfg.retention_seconds=cli.retention_seconds or (3. if cli.mode=='train' else 10.)
    cfg.episode_length_s=max(35.,25.+cfg.retention_seconds)
    cfg.viewer.eye=(.9,.65,1.7);cfg.viewer.lookat=(-.43,.02,1.15)
    cfg.viewer.origin_type='env';cfg.viewer.env_index=0
    env=BoltInsertionEnv(cfg)
    metadata=dict(task=TASK_ID,reset_prepositioned=True,includes_pickup=False,
        installation_complete=False,action_dim=ACTION_DIM,observation_dim=OBSERVATION_DIM,seed=cli.seed,
        force_source='wall_and_keel_contact_normal_plus_friction',
        arguments=vars(cli),env_config=cfg.to_dict(),agent_config=agent_config(cli.device),
        isaaclab_version=importlib.metadata.version('isaaclab'),rsl_rl_version=importlib.metadata.version('rsl-rl-lib'),
        fixture_sha256=hashlib.sha256((ASSETS/'fixture.json').read_bytes()).hexdigest(),
        reset_states_sha256=hashlib.sha256((ASSETS/'reset_states.npz').read_bytes()).hexdigest(),
        sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        physics_check=env.asset_check)
    (out/'configuration.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    env.attach_csv(out/'episodes.csv')
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=None
    if cli.mode!='check':
        runner=OnPolicyRunner(wrapped,agent_config(cli.device),log_dir=str(out) if cli.mode=='train' else None,device=cli.device)
        original_save=runner.save
        def save_with_metadata(path,infos=None):
            info=dict(task=metadata['task'],fixture_sha256=metadata['fixture_sha256'],
                curriculum_level=env.level,reset_prepositioned=True,includes_pickup=False)
            if infos:info.update(infos)
            destination=Path(path)
            temporary=destination.with_name(destination.name+'.tmp')
            original_save(str(temporary),infos=info)
            temporary.replace(destination)
            if cli.mode=='train' and Path(path)!=out/'latest.pt':
                shutil.copyfile(path,out/'latest.tmp.pt');(out/'latest.tmp.pt').replace(out/'latest.pt')
        runner.save=save_with_metadata
        if checkpoint:
            info=runner.load(str(checkpoint),map_location=cli.device)
            if not info or info.get('task')!=metadata['task'] or info.get('fixture_sha256')!=metadata['fixture_sha256']:
                raise RuntimeError('Checkpoint task/assets differ from the current PPO experiment')
            env.level=int(info.get('curriculum_level',cfg.difficulty)) if cli.difficulty=='auto' else cfg.difficulty
        # Compare a known parameter before/after the short smoke training.
        initial=[p.detach().clone() for p in runner.alg.get_policy().parameters()]
    ui=None
    if not cli.headless:
        import omni.ui as ui_module
        ui=ui_module.Window('Nero PPO: single bolt insertion',width=390,height=155)
        with ui.frame:
            with ui_module.VStack():
                label=ui_module.Label('Policy is being trained' if cli.mode=='train' else cli.mode)
                ui_module.Label('Left hand supports keel; right hand is learned.')
                ui_module.Label('Near-hole reset; pickup is not learned yet.')
                status=ui_module.Label('Starting...')
        original=wrapped.step
        def visual_step(actions):
            while app.is_running() and not env.sim.is_playing():
                if env.sim.is_stopped():
                    raise KeyboardInterrupt('Timeline stopped')
                app.update()
            if not app.is_running():raise KeyboardInterrupt('Simulation window closed')
            result=original(actions)
            phases=('settling','PPO insertion','opening','retreating','free retention check')
            status.text=f'Env 0: {phases[int(env.phase[0])]} | depth {float(env.m["depth"][0])*1000:.2f} mm\nAxial {float(env.m["axial_force"][0]):.2f} N | side {float(env.m["lateral_force"][0]):.2f} N\nEpisodes {env.completed_episodes} | difficulty {env.level}'
            return result
        wrapped.step=visual_step
    # Physics/extensions may install their own native handler during scene
    # initialization. Restore Python handling only after the scene and GUI
    # are ready so the training finally block can save the checkpoint.
    signal.signal(signal.SIGINT,signal.default_int_handler)
    print(f'PPO_RUN mode={cli.mode} directory={out}',flush=True)
    interrupted=False
    try:
        if cli.mode=='train':
            try:runner.learn(num_learning_iterations=cli.iterations,init_at_random_ep_len=False)
            except KeyboardInterrupt:interrupted=True
            finally:
                runner.save(str(out/'latest.pt'),infos=dict(task=metadata['task'],fixture_sha256=metadata['fixture_sha256'],
                    curriculum_level=env.level,reset_prepositioned=True,includes_pickup=False))
                if runner.logger.writer is not None:runner.logger.writer.flush();runner.logger.writer.close()
            changed=any(not torch.equal(a,b.detach()) for a,b in zip(initial,runner.alg.get_policy().parameters()))
            result=dict(mode='train',interrupted=interrupted,parameter_update_verified=changed,
                completed_episodes=env.completed_episodes,curriculum_level=env.level,
                checkpoint=str(out/'latest.pt'),installation_complete=False)
        else:
            policy=runner.get_inference_policy(device=cli.device) if runner else None
            trace=[]
            with torch.inference_mode():
                obs=wrapped.get_observations()
                for step in range(cli.max_steps):
                    if not app.is_running():break
                    if policy is not None:action=policy(obs)
                    elif cli.controller=='expert':action=env.expert_action()
                    elif cli.controller=='random':action=torch.randn(env.num_envs,ACTION_DIM,device=env.device)*.4
                    else:action=torch.zeros(env.num_envs,ACTION_DIM,device=env.device)
                    bolt,keel=env._state();p,q=env._ee(1)
                    metrics=env._measure()
                    trace.append(dict(step=step,phase=env.phase[0].item(),bolt=bolt[0].cpu().numpy(),
                        keel=keel[0].cpu().numpy(),tcp=torch.cat((p,q),-1)[0].cpu().numpy(),action=action[0].cpu().numpy(),
                        target=env.hand_targets[1][0].cpu().numpy(),
                        metrics=np.array([float(metrics[key][0]) for key in ('depth','entry','axis_error','slip','gap','hand_force','arm_force',
                            'axial_force','lateral_force','interface_force','contact_load')]),force=env.force_raw[0].cpu().numpy(),
                        reference=float(env.force_reference[0])))
                    obs,reward,done,extras=wrapped.step(action)
                    if env.completed_episodes>=cli.episodes:break
            np.savez_compressed(out/'rollout.npz',step=np.array([r['step'] for r in trace]),
                phase=np.array([r['phase'] for r in trace]),bolt_poses=np.array([r['bolt'] for r in trace]),
                keel_poses=np.array([r['keel'] for r in trace]),tcp_poses=np.array([r['tcp'] for r in trace]),
                actions=np.array([r['action'] for r in trace]),reset_prepositioned=True,includes_pickup=False,
                tcp_targets=np.array([r['target'] for r in trace]),
                metrics=np.array([r['metrics'] for r in trace]),
                metric_names=np.array(['depth_m','entry_radial_m','axis_rad','slip_m','gap_m','hand_force_N','arm_force_N',
                    'axial_force_N','lateral_force_N','interface_force_N','contact_load_N']),
                contact_forces_hole_N=np.array([r['force'] for r in trace]),
                force_references_N=np.array([r['reference'] for r in trace]),
                installation_complete=False,assembly_joints=0,gravity_enabled=True)
            result=dict(mode=cli.mode,controller='policy' if policy else cli.controller,
                requested_episodes=cli.episodes,completed_episodes=env.completed_episodes,
                successful=sum(r['success'] for r in env.records),episodes=env.records,
                rollout_steps=len(trace),budget_exhausted=env.completed_episodes<cli.episodes,
                held_out_reset_states=True,installation_complete=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PPO_DONE '+json.dumps(result if cli.mode=='train' else {k:v for k,v in result.items() if k!='episodes'}),flush=True)
        if cli.mode=='check' and (result['completed_episodes']<cli.episodes or (cli.controller=='expert' and result['successful']<cli.episodes)):
            return 1
        return 0
    finally:wrapped.close()

code=0
try:code=run()
except Exception:
    import traceback
    traceback.print_exc();code=1
finally:
    app.app.post_quit(code);app.close()
raise SystemExit(code)
