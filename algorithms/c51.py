import torch.nn as nn
import torch
from algorithms.DQN_Agent import ReplayMemory
from torch import optim
import numpy as np
import torch.nn.functional as F

class C51Network(nn.Module):
    def __init__(self, num_actions, input_dim, support_size, v_min, v_max, device):
        super(C51Network, self).__init__()
        self.support_size = support_size
        self.num_actions = num_actions
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (support_size - 1)
        self.support = torch.linspace(v_min, v_max, support_size).to(device)

        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions * support_size)
        )

        # Initialize FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.FC(x)
        x = x.view(-1, self.num_actions, self.support_size)
        x = F.softmax(x, dim=2)
        return x

    def get_q_values(self, dist):
        q_values = torch.sum(dist * self.support, dim=2)
        #         print("q_values", q_values)
        return q_values


class C51Agent:
    def __init__(self, env, seed, device, epsilon_max, epsilon_min, epsilon_decay,
                 clip_grad_norm, learning_rate, discount, memory_capacity, support_size, v_min, v_max):
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
        self.device = device

        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity, device)

        self.support_size = support_size
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (support_size - 1)

        self.main_network = C51Network(num_actions=self.action_space.n, input_dim=self.observation_space.n,
                                       support_size=support_size, v_min=v_min, v_max=v_max, device=device).to(device)
        self.target_network = C51Network(num_actions=self.action_space.n, input_dim=self.observation_space.n,
                                         support_size=support_size, v_min=v_min, v_max=v_max, device=device).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if np.random.random() < self.epsilon_max:
            #             print("action_space", self.action_space)
            return self.action_space.sample()
        else:
            with torch.no_grad():
                Q_values = self.main_network.get_q_values(self.main_network(state))
                action = torch.argmax(Q_values).item()
                #                 print(f"State: {state}, Selected action: {action}")
                return action

    def learn(self, batch_size, done):
        if len(self.replay_memory) < batch_size:
            return

        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).float()

        predicted_dist = self.main_network(states).gather(1, actions.unsqueeze(1).expand(-1, -1,
                                                                                         self.support_size)).squeeze(1)

        with torch.no_grad():
            next_dist = self.target_network(next_states)
            next_q_values = self.target_network.get_q_values(next_dist)
            next_actions = next_q_values.max(1)[1]
            #             print("next_actions:", next_actions)
            next_dist = next_dist[range(batch_size), next_actions]

            Tz = rewards + (1 - dones) * self.discount * self.main_network.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(batch_size, self.support_size).to(self.device)
            offset = torch.linspace(0, (batch_size - 1) * self.support_size, batch_size).unsqueeze(1).expand(batch_size,
                                                                                                             self.support_size).to(
                self.device)
            m.view(-1).index_add_(0, (l + offset).view(-1).long(), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1).long(), (next_dist * (b - l.float())).view(-1))


        loss = -torch.sum(m * predicted_dist.log(), dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        #         print(f"Loss: {loss.item()}, Epsilon: {self.epsilon_max}")
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.running_loss += loss.item()
        self.learned_counts += 1
        if done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0

    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        torch.save(self.main_network.state_dict(), path)