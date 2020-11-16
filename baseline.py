#Wrapper function for implemented baselines
def set_hyperparameters(baseline='Procgen'):
  implemented_baselines = {}
  implemented_baselines['Procgen'] = set_hyperparameters_to_baseline_Procgen
  implemented_baselines['PPO'] = set_hyperparameters_to_baseline_PPO

  if baseline not in implemented_baselines.keys():
    raise NotImplementedError("The implemented baselines are: {}".format(implemented_baselines.keys()))
  else:
    implemented_baselines[baseline]()

#Hyperparameters inspired by Procgen Paper (With 32 instead of 64 environments because of memory)
def set_hyperparameters_to_baseline_Procgen():
  global total_steps, num_envs, num_levels, num_steps, num_epochs, batch_size, eps, grad_eps, value_coef, entropy_coef, gamma, lmbda, lr , version
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
  lr= 5e-4
  version = 'Baseline(Procgen)'

  
#Hyperparameters inspired by PPO Paper ( without variable learning rate )
def set_hyperparameters_to_baseline_PPO():
  global total_steps, num_envs, num_levels, num_steps, num_epochs, batch_size, eps, grad_eps, value_coef, entropy_coef, gamma, lmbda, lr, version
  total_steps = 8e6
  num_envs = 32
  num_levels = 10
  num_steps = 128
  num_epochs = 3
  batch_size = 256
  eps = .1
  grad_eps = .5
  value_coef = 1
  entropy_coef = .01
  gamma = 0.99
  lmbda = 0.95
  lr = 2.5e-4
  version = 'Baseline(PPO)'