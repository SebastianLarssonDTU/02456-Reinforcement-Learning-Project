import torch
from utils import make_env, Storage, orthogonal_init, VecFrameStack
from model import NatureEncoder, ImpalaEncoder
from policy import Policy
from datatools import DATA_PATH, MODEL_PATH, create_data_file, add_to_data_file
from my_util import ClippedPPOLoss, ValueFunctionLoss, ClippedValueFunctionLoss
import hyperparameters as h
from datetime import datetime
from pytz import timezone 
import time
import numpy as np

#region init
class PPO():
    def __init__(self,
                print_output=False, 
                file_name=None,
                eval = False,
                eval_cycle=16,
                save_interval = 1e6):
        
        
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
        self.value_clipping = h.value_clipping
        self.death_penalty = h.death_penalty
        self.penalty = h.penalty
        self.save_interval = save_interval
        self.step_start = 0
        self.nstack = h.nstack

        #Create file_name
        self.file_name=self.create_file_name(file_name)

        self.eval = eval
        self.eval_cycle = eval_cycle

        self.print_output= print_output


        #Create Model
        if h.encoder == "Nature":
            self.encoder = NatureEncoder(in_channels = h.in_channels, feature_dim = h.feature_dim)
        elif h.encoder == "Impala":
            self.encoder = ImpalaEncoder(in_channels=h.in_channels, feature_dim = h.feature_dim)      #TODO
        else:
            raise ValueError('Only valid encoders are "Nature" and "Impala"')
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
        
        self.obs_shape = self.env.observation_space.shape
        self.obs_space_shape = (self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]*self.nstack)
        
        return Storage(self.obs_space_shape,
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

        if self.eval:
            create_data_file(self.file_name+'_EVAL' + '.csv')
            #add header
            header = "step,"
            for i in range(self.num_envs):
                header += "env_{}(mean),env_{}(var),".format(i,i)
            header += "avg\n"
            add_to_data_file(header, self.file_name+'_EVAL' + '.csv')

        hyperpar_string = ""
        for key, val in vars(self).items():
            if key in ["encoder", "print_output", "policy", "optimizer", "storage", "env"]:
                continue
            hyperpar_string += "{}, {}\n".format(key, val)
        add_to_data_file(hyperpar_string, self.file_name + '.txt')
        #TODO run through hyperparameters and log them
#endregion
#region training
    def train(self):
        """
             Run training
        """
        
        #INIT LOG
        self.init_log_files()
        
        self.start_time = time.time()
        
        if h.nstack == 1:
            obs = self.env.reset()
        else:
            self.framestack = VecFrameStack(self.env, h.nstack)
            obs = self.framestack.reset()
            obs = torch.from_numpy(obs)
        
        step = self.step_start
        m_counter=1
        
        while step < self.total_steps:
            #If time limit exceeded:
            if self.is_time_spent():
                self.end_training(step)
                return self.policy

            # Use policy to collect data for num_steps steps
            self.run_policy(obs)

            # Optimize policy
            self.optimize_policy()

            #TODO: put in method
            
            #save model every now and then
            if step > self.step_start + m_counter*self.save_interval:
                self.save_policy(self.file_name +"_{}steps".format(self.step_start + step))
                m_counter +=1
            
            # Update stats
            step += self.num_envs * self.num_steps
            if self.print_output:
                print(f'Step: {step}\tMean reward: {self.storage.get_reward()}')
            add_to_data_file("{}, {}\n".format(step, self.storage.get_reward()), self.file_name+'.csv')
            if int((step/(self.num_envs * self.num_steps))%self.eval_cycle) == 0:
                total_reward, all_episode_rewards = self.evaluate_policy(self.num_levels)
                if self.print_output:
                    print("Evaluation done with avg score of {:10f}".format(total_reward))                
                add_to_data_file("{},".format(step), self.file_name+'_EVAL' + '.csv')
                for key in sorted(all_episode_rewards.keys()):
                    add_to_data_file("{:10f}, {:10f},".format(np.mean(all_episode_rewards[key]), np.var(all_episode_rewards[key])), self.file_name+'_EVAL' + '.csv')
                add_to_data_file("{:10f}\n".format(total_reward), self.file_name+'_EVAL' + '.csv')
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
        self.save_policy(self.file_name+"_{}steps".format(self.step_start + last_step))

    def save_policy(self, file_name):
        if self.print_output:
            print("Saved current model in models folder with name {}.pt".format(file_name))
        torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, MODEL_PATH + file_name+'.pt')

    def load_policy(self, file_name):
        checkpoint = torch.load(MODEL_PATH + file_name + '.pt')
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.cuda()

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        
        if self.print_output:
            print("Loaded current model from models folder with name {}.pt".format(file_name))
        
        #save old step count
        if "steps" in file_name:
            self.step_start = int(file_name.split("_")[-1].replace("steps", ""))
        #manually read last step from file
        else:
            f = open(DATA_PATH + file_name +'.csv', "r")
            for last_line in f:
                pass
            f.close()

            last_line = last_line.rstrip() #to remove a trailing newline

            steps, reward = last_line.split(",")
            self.step_start = int(steps)

        #update file_name
        if "steps" in file_name or "loaded" in file_name:
            new_name = ""
            for sub_str in file_name.split("_"):
                if "steps" in sub_str or "loaded" in sub_str:
                    break
                new_name += sub_str+"_"
            file_name = new_name[:-1]
    
        now = datetime.now(timezone('Europe/Copenhagen'))
        self.file_name = file_name + "_loaded_" +now.strftime("%d%b_%Hh%Mm%Ss")

        self.total_steps += self.step_start
            
        return self.policy
    
    def is_time_spent(self):
        time_spent = time.time()-self.start_time
        return time_spent > self.time_limit
    
    def run_policy(self, obs):
        
        self.policy.eval()
        for _ in range(self.num_steps):
            # Use policy
            action, log_prob, value = self.policy.act(obs)
            
            # Take step in environment
            if h.nstack == 1:
                next_obs, reward, done, info = self.env.step(action)
            else:
                _, reward, done, info = self.env.step(action)
                next_obs, _, _, _ = self.framestack.step_wait()
                next_obs = torch.from_numpy(next_obs)
                
            if self.death_penalty:
                reward = reward - self.penalty*done

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
                if self.value_clipping:
                    value_loss = ClippedValueFunctionLoss(value=new_value, sampled_value=b_value, sampled_return=b_returns, clip=self.eps)
                else:
                    value_loss= ValueFunctionLoss(new_value=new_value, old_value= b_value)

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
#endregion
#region evaluation        
    def evaluate_policy(self, 
                        nr_of_levels,
                        print_output=False):
        """
        TODO: Add Video generation
        """
        model = self
        policy = model.policy

        #pick levels we did not train on. 
        eval_env = make_env(model.num_envs, start_level=model.num_levels, num_levels=nr_of_levels)
        if h.nstack == 1:
            obs = self.env.reset()
        else:
            self.framestack = VecFrameStack(self.env, h.nstack)
            obs = self.framestack.reset()
            obs = torch.from_numpy(obs)

        #book-keeping
        completed_envs= []
        counter_compl_envs = np.zeros(model.num_envs)
        episode_rewards = np.zeros(model.num_envs)  #current episode rewards
        rewards = {}
        for i in range(model.num_envs):
            rewards[i] = []
        step_counter = 0

        policy.eval()
        while True:

            # Use policy
            action, log_prob, value = policy.act(obs)

            # Take step in environment
            if h.nstack == 1:
                next_obs, reward, done, info = self.env.step(action)
            else:
                _, reward, done, info = self.env.step(action)
                next_obs, _, _, _ = self.framestack.step_wait()
                next_obs = torch.from_numpy(next_obs)
            
            #if any reward, update envs still not done
            for i in range(len(reward)):
                if reward[i] != 0 and i not in completed_envs:
                    episode_rewards[i] += reward[i]
            
            # If new environment done, complete it
            for i in [index for index in range(len(done)) if done[index] == True]:
                if i not in completed_envs:
                    counter_compl_envs[i] += 1
                    if print_output:
                        print("Environment {:2d} completed its {:4d}th level at timestep {:6d} with a reward of {:10f}".format(i, int(counter_compl_envs[i]), step_counter, episode_rewards[i]))
                    rewards[i].append(episode_rewards[i])
                    episode_rewards[i] = 0
                    if counter_compl_envs[i] == nr_of_levels:
                        completed_envs.append(i)  
                
        

            # If all environments are done, break
            if len(completed_envs) == model.num_envs:
                break
            step_counter +=1
        # end while
        
        # Calculate average return
        total_reward = []
        for key, value in rewards.items():
            total_reward.append(sum(value))
        total_reward = np.mean(total_reward)/nr_of_levels

        if print_output:
            print('Average return:', total_reward)

        policy.train()

        return total_reward, rewards
#endregion
