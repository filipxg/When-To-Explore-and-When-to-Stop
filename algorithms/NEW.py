from algorithms.IDS import IDS_Agent
import torch

class NewAlg_Agent(IDS_Agent):
    def select_action(self, state):
        """
        Selects an action using new agent strategy OR based on the Q-values.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """

        with torch.no_grad():
            q_values = torch.stack(tuple(model(state) for model in self.models))

            means = torch.mean(q_values, dim=0)
            std = torch.std(q_values, dim=0)
            var = torch.var(q_values, dim=0)

            a_star = torch.argmax(means + 3 * std).item()
            q_star = means[a_star]
            var_star = var[a_star]

            numerator = (q_star - means) ** 2
            denominator = var_star + var + 1e-7

            score = numerator / denominator

            action = torch.argmin(score).item()

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
                state_table[row, column] = self.select_action(onehot_vector)
        return state_table