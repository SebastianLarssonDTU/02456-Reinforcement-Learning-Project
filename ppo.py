import torch
from my_project.utils import make_env, Storage, orthogonal_init
from my_project.model import Encoder
from my_project.policy import Policy
from my_project.datatools import DATA_PATH, MODEL_PATH, create_data_file, add_to_data_file, ClippedPPOLoss, ValueFunctionLoss
from datetime import datetime
from pytz import timezone 


class PPO():
    def __init__(self,
                print_output=False, 
                file_name=None,
                total_steps = 8e6,
                num_envs = 32,
                num_levels = 10,
                num_steps = 256,
                num_epochs = 3,
                batch_size = 512,
                eps = .2,
                grad_eps = .5,
                value_coef = .5,
                entropy_coef = .01,
                lr=5e-4,
                opt_extra = 1e-5,
                gamma=0.99,
                lmbda = 0.95, 
                version = '',
                optimizer = torch.optim.Adam,
                in_channels = 3,
                feature_dim = 512):
        
        #Save parameters
        self.print_output= print_output
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.num_levels = num_levels
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps = eps
        self.grad_eps = grad_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.version = version

        #Create file_name
        self.file_name=self.create_file_name(file_name)
        
        #Create Model
        self.encoder = Encoder(in_channels = in_channels, feature_dim = feature_dim)
        self.policy = Policy(encoder = self.encoder, feature_dim = feature_dim, num_actions = 15)
        self.policy.cuda()
        self.optimizer = optimizer(self.policy.parameters, lr, opt_extra)
        self.env = make_env(num_envs, num_levels=num_levels)

        #print
        if print_output:
            print('Observation space:', self.env.observation_space)
            print('Action space:', self.env.action_space.n)

        # Define temporary storage
        self.storage = self.create_storage()

        #INIT LOG
        self.init_log_files()

    
    
    
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
        add_to_data_file("Step, Mean reward\n", self.file_name+'csv')
        create_data_file(self.file_name + 'txt')
        add_to_data_file("Parameter name, Value\n", self.file_name+'txt')

    def train(self):
        """
             Run training
        """
        obs = self.env.reset()
        step = 0
        while step < self.total_steps:
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
        torch.save(self.policy.state_dict(), MODEL_PATH + self.file_name+'.pt')


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
