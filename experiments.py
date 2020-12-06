import hyperparameters as h
from baseline import set_hyperparameters
from ppo import PPO

#Experiment with different batch sizes?

def print_list_of_experiments():
    for i in range(9):
        run_experiment(i, run=False)


def run_experiment(input, par=None, run=True, levels=10, load_model=False, path=None, save_interval=1e6):

    #Set hyperparameters
    if input == 0:
        #Baseline PPO (without value clipping)
        set_hyperparameters(baseline="PPO")
        description = "Baseline inspired by PPO article"
        h.version = "Experiment0"

    elif input == 1:
        #Baseline Procgen (without value clipping)
        set_hyperparameters(baseline="Procgen")
        description = "Baseline inspired by Procgen article"
        h.version = "Experiment1"

    elif input == 2:
        #PPO with value clipping
        set_hyperparameters(baseline="PPO")
        description = "Modified PPO baseline with value clipping enabled"
        h.value_clipping = True
        h.version = "Experiment2"

    elif input == 3:
        #PPO, with value clipping, changed learning rate
        set_hyperparameters(baseline="PPO")
        if par is None:
            if run:
                raise ValueError("Needs new learning rate given to 'par' ")
            else:
                par = "given by par variable"
        else:
            h.lr = par
        description = "Modified PPO baseline with value clipping enabled and learning rate {}".format(par)
        h.value_clipping = True
        h.version = "Experiment3"

    elif input == 4:
        #custom penalty for death
        set_hyperparameters(baseline='PPO')
        description = "Modified PPO baseline with value clipping enabled and reward penalty on death (1 as default)"
        h.value_clipping = True
        h.death_penalty = True
        if par is not None:
            h.penalty = par
        else:
            h.penalty = 1

        h.version = "Experiment4"

    elif input == 5:
        #Impala encoder with hyperparameters inspired by Impala paper
        set_hyperparameters(baseline="Impala")
        description = "Baseline inspired by IMPALA paper (No value clipping)"
        h.version = "Experiment5"

    elif input == 6:
        #Impala encoder with hyperparameters inspired by Impala paper, and with value clipping
        set_hyperparameters(baseline="Impala")
        description = "Inspired by IMPALA paper (With value clipping)"
        h.value_clipping = True
        h.version = "Experiment6"

    elif input == 7:
        #Mix of Impala architecture and procgen hyperparameters
        set_hyperparameters(baseline="Impala")
        description = "Inspired by both IMPALA and Procgen papers (With value clipping)"
        h.value_clipping = True
        h.batch_size = 512
        h.version = "Experiment7"
    elif input == 8:
        #experiment 7 with death penalty
        set_hyperparameters(baseline="Impala")
        description = "Inspired by both IMPALA and Procgen papers (With value clipping) and added death penalty"
        h.value_clipping = True
        h.batch_size = 512
        h.death_penalty = True
        if par is not None:
            h.penalty = par
        else:
            h.penalty = 1
        h.version = "Experiment8"
    elif input == 9:
        #Impala encoder with hyperparameters inspired by Impala paper
        set_hyperparameters(baseline="Impala")
        description = "Testing frame stacking with impala"
        h.version = "Experiment9"
        h.nstack = 4
    else:
        raise ValueError("Only experiment 0-8 is defined")

    h.num_levels = levels
    h.version = h.version +"_{}levels".format(levels)

    print("***** Experiment {} *****".format(input))
    print("Description:    " +description)
    if run:
        #Create Model
        model = PPO(print_output=True, eval=True, save_interval=save_interval)
        
        #Train
        model.train()
