import hyperparameters as h
from baseline import set_hyperparameters
from ppo import PPO

#Experiment with different batch sizes?

def print_list_of_experiments():
    for i in range(8):
        run_experiment(i, run=False)


def run_experiment(input, par=None, run=True):

    #Set hyperparameters
    if input == 0:
        #Baseline PPO (without value clipping)
        set_hyperparameters(baseline="PPO")
        description = "Baseline inspired by PPO article"

    elif input == 1:
        #Baseline Procgen (without value clipping)
        set_hyperparameters(baseline="Procgen")
        description = "Baseline inspired by Procgen article"

    elif input == 2:
        #Procgen with value clipping
        set_hyperparameters(baseline="Procgen")
        description = "Modified Procgen baseline with value clipping enabled"
        h.value_clipping = True

    elif input == 3:
        #Procgen, with value clipping, changed learning rate
        set_hyperparameters(baseline="Procgen")
        if par is None:
            if run:
                raise ValueError("Needs new learning rate given to 'par' ")
            else:
                par = "given by par variable"
        else:
            h.lr = par
        description = "Modified Procgen baseline with value clipping enabled and learning rate {}".format(par)
        h.value_clipping = True

    elif input == 4:
        #custom penalty for death
        set_hyperparameters(baseline='Procgen')
        description = "Modified Procgen baseline with value clipping enabled and reward penalty of -1 on death"
        h.value_clipping = True
        h.death_penalty = True
        #TODO: Implement

    elif input == 5:
        #Impala encoder with hyperparameters inspired by Impala paper
        set_hyperparameters(baseline="Impala")
        description = "Baseline inspired by IMPALA paper (No value clipping)"

    elif input == 6:
        #Impala encoder with hyperparameters inspired by Impala paper, and with value clipping
        set_hyperparameters(baseline="Impala")
        description = "Inspired by IMPALA paper (With value clipping)"
        h.value_clipping = True

    elif input == 7:
        #Mix of Impala architecture and procgen hyperparameters
        set_hyperparameters(baseline="Impala")
        description = "Inspired by both IMPALA and Procgen papers (With value clipping)"
        h.value_clipping = True
        h.batch_size = 512

    else:
        raise ValueError("Only experiment 0-7 is defined")

    print("***** Experiment {} *****".format(input))
    print("Description:    " +description)
    if run:
        #Create Model
        model = PPO(print_output=True, eval=True)
        
        #Train
        model.train()