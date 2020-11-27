import torch
#Default is set to same as procgen
total_steps = 8e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
gamma = 0.999
lmbda = 0.95
optimizer=torch.optim.Adam
lr= 5e-4
opt_extra = 1e-5
version = 'Default'
in_channels = 3
feature_dim = 512
time_limit_hours = 12
time_limit_minutes = 0
time_limit_seconds = 0
value_clipping = False
death_penalty = False
encoder = "Nature"
penalty = 1