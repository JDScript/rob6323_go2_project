# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

- After logging into Greene, `cd` into your home directory and clone your fork into $HOME only, not elsewhere.  
```
cd $HOME
git clone rob6323_go2_project
```
This ensures subsequent scripts and paths resolve correctly on the cluster and during grading.

## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py  

## How to edit

- Option A: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option B: Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code).  
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

- From your laptop, recursively copy a run directory using the DTN endpoint for reliable transfers to and from Greene.  
```
scp -r <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./logs
```
Using dtn.hpc.nyu.edu is recommended for large file transfers such as videos and event files.

## Visualize with TensorBoard
If you wish, you can visualize the training metrics of your experiments the following way:
- Install TensorBoard locally, start a tensorboard server at your downloaded logs directory, and open your browser at the local tensorboard webpage to inspect training metrics and learning curves.
On your computer (not on greene!):
```
pip install tensorboard
tensorboard --logdir ./logs
```
then open the provided localhost URL in your browser.
TensorBoard will load the events files from your runs and display scalars, histograms, and other summaries for your training and evaluation phases.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

---
Students should only edit README.md below this ligne.
