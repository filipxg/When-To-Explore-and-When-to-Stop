import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from algorithms.DQN_Agent import DQN_Network, ReplayMemory


class UCB_Agent:
    """
    UCB Agent Class. This class defines some key elements of the UCB algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """

    def __init__(self, env, seed, device, epsilon_max, epsilon_min, epsilon_decay,
                 clip_grad_norm, learning_rate, discount, memory_capacity):

        self.device = device

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)  # Set the seed to get reproducible results when sampling the action space
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity, device)

        # Initiate the network models
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=self.observation_space.n).to(
            device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.env = env
        self.aleatoric_map = env.get_aleatoric_uncertainty()

        self.clip_grad_norm = clip_grad_norm  # For clipping exploding gradients caused by high reward value
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

        self.state_action_counts = {}
        self.state_counts = {}
        self.c = 5.0

    def select_action(self, state, training=True, use_aleatoric=True):
        """
        Selects an action using the UCB algorithm OR based on the Q-values.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """

        if (training):
            C = self.c
        else:
            C = 0

        aleatoric = self.aleatoric_map[torch.argmax(state).item()]
        aleatoric /= (torch.sum(aleatoric) + 1e-5)  # Normalise

        state_idx = state.argmax().item()

        if state_idx not in self.state_counts:
            self.state_counts[state_idx] = 0
            self.state_action_counts[state_idx] = np.zeros(self.action_space.n)

        # Forward pass through the main network to find the Q-values of the state
        with torch.no_grad():
            Q_values = self.main_network(state)

        # Compute UCB values for each action
        N_s = self.state_counts[state_idx] + 1
        ucb_values = np.zeros(self.action_space.n)
        for a in range(self.action_space.n):
            N_s_a = self.state_action_counts[state_idx][a] + 1
            if use_aleatoric:
                # Use aleatoric uncertainty
                ucb_values[a] = Q_values[a].item() + C * (np.sqrt(aleatoric[a]) + 1) * np.sqrt(np.log(N_s) / N_s_a)
            else:
                # No aleatoric uncertainty used
                ucb_values[a] = Q_values[a].item() + C * np.sqrt(np.log(N_s) / N_s_a)

        # Select action with the maximum UCB value
        action = np.argmax(ucb_values)

        # Update counts
        if (training):
            self.state_action_counts[state_idx][action] += 1
            self.state_counts[state_idx] += 1

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
                state_table[row, column] = self.select_action(onehot_vector, training=False)
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

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        predicted_q = self.main_network(
            states)  # forward pass through the main network to find the Q-values of the states
        predicted_q = predicted_q.gather(dim=1,
                                         index=actions)  # selecting the Q-values of the actions that were actually taken

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[
                0]  # not argmax (because we want the maxmimum q-value, not the action that maximize it)

        next_target_q_value[dones] = 0  # Set the Q-value for terminal states to zero
        y_js = rewards + (self.discount * next_target_q_value)  # Compute the target Q-values
        loss = self.critertion(predicted_q, y_js)  # Compute the loss

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1
        if done:
            episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
            self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass and update the gradients

        # # Uncomment the following two lines to find the best value for clipping gradient (Comment torch.nn.utils.clip_grad_norm_ while uncommenting the following two lines)
        # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), float('inf'))
        # print("Gradient norm before clipping:", grad_norm_before_clip)

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def hard_update(self):
        """
        Naive update: Update the target network parameters by directly copying
        the parameters from the main network.
        """

        self.target_network.load_state_dict(self.main_network.state_dict())

    def soft_update(self, tau):
        """
        Soft update: Update the target network parameters by blending the parameters
        from the main network and the target network.
        """
        for target_param, main_param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

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
        torch.save(self.main_network.state_dict(), path)