"""Export TensorBoard PPO curves to a PNG; runs without starting Isaac Sim."""
import argparse
import os
from ppo_assembly.paths import ROOT,local_path
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir')
args=parser.parse_args();directory=local_path(args.run_dir)
events=EventAccumulator(str(directory));events.Reload()
keys=events.Tags()['scalars']
panels=[('Reward',['Train/mean_reward']),('Success',['Task/recent_success_rate','Episode/Task/recent_success_rate']),
    ('Policy loss',['Loss/surrogate','Loss/policy']),('Value loss',['Loss/value_function','Loss/value']),
    ('Entropy / exploration',['Policy/mean_noise_std','Policy/mean_std']),
    ('Insertion',['Task/depth_mm','Task/radial_error_mm','Episode/Task/depth_mm','Episode/Task/radial_error_mm']),
    ('Insertion contact force (N)',['Force/axial_N','Force/lateral_N','Force/reference_N']),
    ('Contact peaks (N)',['Force/step_peak_N']),('Force reward cost',['Reward/force_cost'])]
fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
for ax,(title,wanted) in zip(axes.flat,panels):
    selected=[k for k in wanted if k in keys]
    for k in selected:
        data=events.Scalars(k);ax.plot([r.step for r in data],[r.value for r in data],label=k)
    ax.set(title=title,xlabel='PPO iteration');ax.grid(alpha=.25)
    if selected:ax.legend(fontsize=7)
    else:ax.text(.5,.5,'No samples yet',ha='center',transform=ax.transAxes)
fig.suptitle('Nero PPO training: '+directory.name)
output=directory/'training_curves.png';fig.savefig(output,dpi=150);plt.close(fig)
print(output)
print('Available scalar tags:',', '.join(keys))
