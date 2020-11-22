import torch
from utils import make_env, Storage, orthogonal_init
from model import Encoder
from policy import Policy
from datatools import DATA_PATH, MODEL_PATH, create_data_file, add_to_data_file
from my_util import ClippedPPOLoss, ValueFunctionLoss
import hyperparameters as h
from datetime import datetime
from pytz import timezone 
import time


class PPO():
    def __init__(self,
                print_output=False, 
                file_name=None):
        
        
        #Save parameters from hyperparameters module
        self.total_steps = h.total_steps
        self.num_envs = h.num_envs
        self.num_levels = h.num_levels
        self.num_steps = h.num_steps
        self.num_epochs = h.num_epochs
        self.batch_size = h.batch_size
        self.eps = h.eps
        self.grad_eps = h.grad_eps
        self.value_coef = h.value_coef
        self.entropy_coef = h.entropy_coef
        self.lr = h.lr
        self.gamma = h.gamma
        self.lmbda = h.lmbda
        self.version = h.version
        self.time_limit = 60*60*h.time_limit_hours + 60*h.time_limit_minutes + h.time_limit_seconds

        #Create file_name
        self.file_name=self.create_file_name(file_name)

        #INIT LOG
        self.init_log_files()

        self.print_output= print_output


        #Create Model
        self.encoder = Encoder(in_channels = h.in_channels, feature_dim = h.feature_dim)
        self.policy = Policy(encoder = self.encoder, feature_dim = h.feature_dim, num_actions = 15)
        self.policy.cuda()
        self.optimizer = h.optimizer(self.policy.parameters(), lr=self.lr, eps=h.opt_extra)
        self.env = make_env(self.num_envs, num_levels=self.num_levels)

        #print
        if print_output:
            print('Observation space:', self.env.observation_space)
            print('Action space:', self.env.action_space.n)

        # Define temporary storage
        self.storage = self.create_storage()

    
    
    
    def create_storage(self):
        return Storage(self.env.observation_space.shape,
                       self.num_steps,
                       self.num_envs,
                       gamma = self.gamma,
                       lmbda = self.lmbda)

    def create_file_name(self, file_name):
        if file_name is not None:
            return file_name
        else:
            now = datetime.now(timezone('Europe/Copenhagen'))
            return self.version+'_Run_' + now.strftime("%d%b_%Hh%Mm%Ss")

    def init_log_files(self):
        create_data_file(self.file_name + '.csv')
        add_to_data_file("Step, Mean reward\n", self.file_name+'.csv')
        create_data_file(self.file_name + '.txt')
        add_to_data_file("Parameter name, Value\n", self.file_name+'.txt')

        hyperpar_string = ""
        for key, val in vars(self).items():
            hyperpar_string += "{}, {}\n".format(key, val)
        add_to_data_file(hyperpar_string, self.file_name + '.txt')
        #TODO run through hyperparameters and log them

    def train(self):
        """
             Run training
        """
        self.start_time = time.time()
        
        obs = self.env.reset()
        step = 0
        while step < self.total_steps:
            #If time limit exceeded:
            if self.is_time_spent():
                self.end_training(step)
                return self.policy

            # Use policy to collect data for num_steps steps
            self.run_policy(obs)

            # Optimize policy
            self.optimize_policy()

            # Update stats
            step += self.num_envs * self.num_steps
            if self.print_output:
                print(f'Step: {step}\tMean reward: {self.storage.get_reward()}')
            add_to_data_file("{}, {}\n".format(step, self.storage.get_reward()), self.file_name+'.csv')
        #end while loop

        if self.print_output:
            print('Completed training!')
        self.end_training(step)
        return self.policy


    def end_training(self, last_step):
      #Add to log file
      add_to_data_file('Time spent (in seconds), {:.2f}\n'.format(time.time()-self.start_time) + \
                        "Steps taken, {}\n".format(last_step) + \
                        "Done, False\n", 
                        self.file_name + '.txt')
      torch.save(self.policy.state_dict(), MODEL_PATH + self.file_name+'.pt')
      
      #Add environment to a csv file
      env_text = ''
      for x in self.storage.current_env():
        env_text += str(x)
        env_text += ','
      print(env_text)  
      create_data_file(self.file_name + '_env.txt')
      add_to_data_file(env_text, self.file_name +'_env.txt')
    
    def is_time_spent(self):
        time_spent = time.time()-self.start_time
        return time_spent > self.time_limit
    
    def run_policy(self, obs):
        
        self.policy.eval()
        for _ in range(self.num_steps):
            # Use policy
            action, log_prob, value = self.policy.act(obs)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)

            # Store data
            self.storage.store(obs, action, reward, done, info, log_prob, value)
            
            # Update current observation
            obs = next_obs

            # Add the last observation to collected data
            _, _, value = self.policy.act(obs)
            self.storage.store_last(obs, value)

            # Compute return and advantage
            self.storage.compute_return_advantage()

    
    def optimize_policy(self):
        # Optimize policy
        self.policy.train()
        for _ in range(self.num_epochs):

            # Iterate over batches of transitions
            generator = self.storage.get_generator(self.batch_size)
            for batch in generator:
                #Results from using old policy on environment
                b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

                # Get current policy outputs
                new_dist, new_value = self.policy(b_obs)
                new_log_prob = new_dist.log_prob(b_action)

                # Clipped policy objective
                pi_loss = ClippedPPOLoss(advantage=b_advantage, 
                                        log_pi=new_log_prob, 
                                        log_old_pi=b_log_prob, 
                                        eps=self.eps)


                # # Clipped value function objective
                # #Assume value_loss = ClippedValueFunctionLoss 
                value_loss= ValueFunctionLoss(new_value=new_value, 
                                            old_value= b_value)

                # Entropy loss
                entropy_loss = new_dist.entropy().mean()

                # Backpropagate losses
                loss = -(pi_loss - self.value_coef * value_loss + self.entropy_coef*entropy_loss)
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_eps)

                # Update policy
                self.optimizer.step()
                self.optimizer.zero_grad()
