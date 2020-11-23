# 02456-Reinforcement-Learning-Project

Command to run environment interactively:
python -m procgen.interactive --env-name starpilot --distribution-mode easy >>D:Dropbox/DTU/Logs/test_output.txt

## Relevant Articles and Git Repos
Reinforcement Learning with Augmented Data:
[Article](https://arxiv.org/abs/2004.14990), [Git](https://github.com/MishaLaskin/rad)

IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures:
[Article](https://arxiv.org/pdf/1802.01561.pdf), [Git](https://github.com/deepmind/scalable_agent)

Leveraging Procedural Generation to Benchmark Reinforcement Learning:
[Article](https://arxiv.org/pdf/1912.01588.pdf), [Git](https://github.com/openai/procgen)

Proximal Policy Optimization Algorithms:
[Article](https://arxiv.org/pdf/1707.06347.pdf)


## Brainstorm ideas
Pre-evaluation method CLEAN UP!

Evaluation method
  - Stop after x games
  - implement it such that we can evaluate during training (every 100k steps or something)
 
Value clipping on/off

Tweak non-conventional hyperparameters (the hyperparameters that are not necessarily part of the model), i.e. value coefficient or...
  - We argue that external sources have already searched for optimal hyperparameters (Procgen paper and PPO paper), and we therefore trust that their hyperparameters are pseudo optimal.
  - We choose not to focus on entropy regularisation because exploration is not very important in Starpilot

Impala implementation without LSTM / RNN
  - Used in Procgen paper

Frame stacking
  - Procgen chose not to use frame stacking, as they argue it only has a minimal effect

Data augmentation



## Testing Phase 1
* Run baseline PPO and Procgen again (and clean folder)
* Run Procgen with value clipping on
* Batchsize, number of steps, lr, optimizer, epsilon, value coeff

Find optimal hyperparameters and continue with those
(TEST) More levels, hard, more steps

## Testing Phase 2 (Impala)
* Run Impala baseline (with their parameters)
* Talk later...

## Testing Phase 3
