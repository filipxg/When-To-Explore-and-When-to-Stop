import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.DQN_Agent import DQN_Network, ReplayMemory


class IDS_Agent:
    """
    IDS Agent Class. This class defines some key elements of the IDS algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """
    
    def __init__(self, seed, device, env, epsilon_max, epsilon_min, epsilon_decay, 
                  clip_grad_norm, learning_rate, discount, memory_capacity, num_ensembles):
        
        self.device = device
        
        # To save the history of network loss
        self.loss_history = [[] for _ in range(num_ensembles)]
        self.running_loss = [0 for _ in range(num_ensembles)]
        self.learned_counts = [0 for _ in range(num_ensembles)]
                     
        # RL hyperparameters
        self.epsilon_max   = epsilon_max
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount      = discount
        self.env = env
        self.aleatoric_map = env.get_aleatoric_uncertainty()

        self.action_space  = env.action_space
        self.action_space.seed(seed) # Set the seed to get reproducible results when sampling the action space 
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity, device)
        
        # Initiate the network models
        self.models = [DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device) for _ in range(num_ensembles)]
        self.targets = [DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device).eval() for _ in range(num_ensembles)]
        for target, main in zip(self.targets, self.models):
            target.load_state_dict(main.state_dict())
            
        self.clip_grad_norm = clip_grad_norm # For clipping exploding gradients caused by high reward value
        self.critertion = [nn.MSELoss() for _ in range(num_ensembles)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
                

    def select_action(self, state, greedy=False):
        """
        Selects an action using epsilon-greedy strategy OR based on the Q-values.
        
        Parameters:
            state (torch.Tensor): Input tensor representing the state.
        
        Returns:
            action (int): The selected action.
        """
        
        # Exploitation: the action is selected based on the Q-values.    
        with torch.no_grad():
            q_values = torch.stack(tuple(model(state) for model in self.models))

            
            means = torch.mean(q_values,dim=0)
            std = torch.std(q_values,dim=0)
            var = torch.var(q_values,dim=0)

            if greedy:
                return torch.argmax(means).item()
                            
            ub = means + 3*std
            lb = means - 3*std
            
            a_star = torch.argmax(ub).item()
            ub_star = torch.max(ub)
            
            regret = ub_star - lb

            # NOT USED
            aleatoric = self.aleatoric_map[torch.argmax(state).item()]
            aleatoric /= (torch.mean(aleatoric) + 1e-5) # Normalise

            info_gain = var + 1e-5
            
            ids = (regret**2)/info_gain
            
            # ids[a_star] = float("inf")
            
            action = torch.argmin(ids).item()

            return action
   
    def get_best_actions(self, state_table, device):
        """
        parameter state_table: a table with the same dimensions as the map the agent trained on

        returns: Tabel of the optimal action at each state
        """
        for row in range(len(state_table)):
            for column in range(len(state_table[0])):
                index = row * (len(state_table[0])) + column
                onehot_vector = torch.zeros(len(state_table) * len(state_table[0]), dtype=torch.float32, device=device)
                onehot_vector[index] = 1
                state_table[row, column] = self.select_action(onehot_vector, greedy=True)
        return state_table

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.
        
        Parameters:
            batch_size (int): The number of experiences to sample from the replay memory.
            done (bool): Indicates whether the episode is done or not. If done,
            calculate the loss of the episode and append it in a list for plot.
        """ 
        
        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
         
        # # Preprocess the data for training
        actions       = actions.unsqueeze(1)
        rewards       = rewards.unsqueeze(1)
        dones         = dones.unsqueeze(1)       
        
        
        # # The following prints are for debugging. Use them to indicate the correct shape of the tensors.
        # print()
        # print('After--------After')
        # print("states:", states.shape)
        # print("actions:", actions.shape)
        # print("next_states:", next_states.shape)
        # print("rewards:", rewards.shape)
        # print("dones:", dones.shape)
        
        for model, target, optimizer, index, critertion in zip(self.models, self.targets, self.optimizers, range(len(self.models)), self.critertion):
            
            predicted_q = model(states) # forward pass through the main network to find the Q-values of the states
            predicted_q = predicted_q.gather(dim=1, index=actions) # selecting the Q-values of the actions that were actually taken
    
            # Compute the maximum Q-value for the next states using the target network
            with torch.no_grad():            
                next_target_q_value = target(next_states).max(dim=1, keepdim=True)[0] # not argmax (cause we want the maxmimum q-value, not the action that maximize it)
                
            
            next_target_q_value[dones] = 0 # Set the Q-value for terminal states to zero
            y_js = rewards + (self.discount * next_target_q_value) # Compute the target Q-values
            loss = critertion(predicted_q, y_js) # Compute the loss
            
            # Update the running loss and learned counts for logging and plotting
            self.running_loss[index] += loss.item()
            self.learned_counts[index] += 1
    
            if done:
                episode_loss = self.running_loss[index] / self.learned_counts[index] # The average loss for the episode
                # print(episode_loss)
                self.loss_history[index].append(episode_loss) # Append the episode loss to the loss history for plotting
                # Reset the running loss and learned counts
                self.running_loss[index] = 0
                self.learned_counts[index] = 0
                
            optimizer.zero_grad() # Zero the gradients
            loss.backward() # Perform backward pass and update the gradients
            
            # # Uncomment the following two lines to find the best value for clipping gradient (Comment torch.nn.utils.clip_grad_norm_ while uncommenting the following two lines)
            # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), float('inf'))
            # print("Gradient norm before clipping:", grad_norm_before_clip)
            
            # Clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            
            optimizer.step() # Update the parameters of the main network using the optimizer
 

    def hard_update(self):
        """
        Navie update: Update the target network parameters by directly copying 
        the parameters from the main network.
        """
        
        for model, target in zip(self.models, self.targets):
            target.load_state_dict(model.state_dict())

    
    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.
        
        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """
        
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)
        

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.
        
        """
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path+str(i)+'.pth')

    def load(self, path):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(path+str(i)+'.pth'))
            model.eval()
                  