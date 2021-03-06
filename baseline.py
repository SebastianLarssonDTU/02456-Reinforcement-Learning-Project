import hyperparameters as h
#Wrapper function for implemented baselines
def set_hyperparameters(baseline='Procgen'):
  implemented_baselines = {}
  implemented_baselines['Procgen'] = set_hyperparameters_to_baseline_Procgen
  implemented_baselines['PPO'] = set_hyperparameters_to_baseline_PPO
  implemented_baselines['Impala'] = set_hyperparameters_to_baseline_Impala

  if baseline not in implemented_baselines.keys():
    raise NotImplementedError("The implemented baselines are: {}".format(implemented_baselines.keys()))
  else:
    implemented_baselines[baseline]()

#Hyperparameters inspired by Procgen Paper (With 32 instead of 64 environments because of memory)
def set_hyperparameters_to_baseline_Procgen():
  h.total_steps = 8e6
  h.num_envs = 32
  h.num_levels = 10
  h.num_steps = 256
  h.num_epochs = 3
  h.batch_size = 512
  h.eps = .2
  h.grad_eps = .5
  h.value_coef = .5
  h.entropy_coef = .01
  h.gamma = 0.999
  h.lmbda = 0.95
  h.lr= 5e-4
  h.version = 'Baseline(Procgen)'
  h.value_clipping = False
  h.death_penalty = False
  h.encoder = "Nature"

  
#Hyperparameters inspired by PPO Paper ( without variable learning rate )
def set_hyperparameters_to_baseline_PPO():
  h.total_steps = 8e6
  h.num_envs = 32
  h.num_levels = 10
  h.num_steps = 256 #originally 128 in PPO paper (we want it to be comparable to Procgen)
  h.num_epochs = 3
  h.batch_size = 512 #originally 256 in PPO paper (we want it to be comparable to Procgen)
  h.eps = .1
  h.grad_eps = .5
  h.value_coef = 1
  h.entropy_coef = .01
  h.gamma = 0.99
  h.lmbda = 0.95
  h.lr = 2.5e-4
  h.version = 'Baseline(PPO)'
  h.value_clipping = False  
  h.death_penalty = False
  h.encoder = "Nature"

def set_hyperparameters_to_baseline_Impala():
  h.total_steps = 8e6
  h.num_envs = 32
  h.num_levels = 10
  h.num_steps = 256
  h.num_epochs = 3
  h.batch_size = 32 #Better with 512?
  h.eps = .2
  h.grad_eps = .5
  h.value_coef = .5
  h.entropy_coef = .01
  h.gamma = 0.99
  h.lmbda = 0.95
  h.lr= 0.0006
  h.version = 'Baseline(Impala)'
  h.value_clipping = False
  h.death_penalty = False
  h.encoder = "Impala"