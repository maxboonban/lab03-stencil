import numpy as np
import random
from collusion_env import CollusionEnv
class CollusionLearner():
    def __init__(self, num_possible_actions, num_possible_states):
        grid_lower_bound = 1.45  # lowest discrete price
        grid_upper_bound = 1.95  # highest discrete price
        self.num_actions  = num_possible_actions  # number of discrete prices (seller's action space)
        self.num_states = num_possible_states

        self.learning_rate = 0.15
        self.discount_factor = 0.95
        self.beta = 0.00001
        self.eps = []
        self.q = np.zeros((num_possible_actions, num_possible_states))
        self.action_price_space = np.linspace(
            grid_lower_bound, grid_upper_bound, self.num_actions)  # price grid
        self.a_i = 2  # product quality index (assumed to be equal for all sellers)
        self.c_i = 1  # seller cost (assumed to be equal for all sellers)
        self.a_0 = 0  # quality of outside option
        self.mu = 0.25  # horizontal differentiation parameter
        self.max_steps = 1000000  # control how long simulation takes to run
        
        self.init_q()
        self.my_action_hist = []
        self.opp_action_hist = []
    
    def determine_state(self):
        """Generates a state representation from your agent's and your opponent's action histories."""
        my_action_hist = self.my_action_hist
        opp_action_hist = self.opp_action_hist
        NUM_ACTIONS = self.num_actions
        # TODO: Fill out your state representation
        # No state-space discretization is needed since it's already being discretized

        if not my_action_hist or not opp_action_hist:
            return 0
        
        my_last_action = my_action_hist[-1]
        opp_last_action = opp_action_hist[-1]

        # Put each discrete action from myself and opponent into its own state-space
        return my_last_action + self.num_actions * opp_last_action
        # raise NotImplementedError
        
    def check_if_converged(self, old_optimal_actions, step):
        """Checks if Q matrices have converged, i.e., if optimal actions do not change for 100000 steps"""
        observation = self.determine_state()
        new_optimal_actions = [np.argmax(self.q[:, observation])]
        action_diff = np.sum(np.absolute(
            np.array(new_optimal_actions) - np.array(old_optimal_actions)))
        q_table_stable = True if action_diff == 0 else False

        has_converged = False
        if step == 1:
            self.convergence_counter = 0
        if q_table_stable:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        if self.convergence_counter == 100000:
            print("Converged: True")
            has_converged = True
            print("Convergence_counter:", self.convergence_counter)
            print("Done: ", has_converged)
            print("-------------------------\n\n\n")

        return has_converged

    def logit_demand_func(self, price_i, price_j): 
        a_i = self.a_i # Product quality index
        a_0 = self.a_0 # Quality of outside option
        mu = self.mu # Horizontal differentiation parameter
        price_i = price_i # Your price
        prices = np.array([price_i, price_j]) # Array of prices (i -> your price), (j -> opp price)
        
        return np.exp((a_i - price_i) / mu) / (np.sum(np.exp((a_i - prices) / mu)) + np.exp(a_0 / mu))

    def init_q(self):
        """Used to initialize Q tables"""
        for s in range(self.num_states): 
            for a1 in range(self.num_actions): 
                reward = 0
                for a2 in range(self.num_actions):
                    price_i = self.action_price_space[a1]
                    price_j = self.action_price_space[a2]
                    demand_0 = self.logit_demand_func(price_i, price_j)
                    profit_0 = (price_i - self.c_i) * demand_0
                    # Assume all actions by the other agent are uniformly possible
                    reward += profit_0 / self.num_actions
                self.q[a1][s] = float(reward / (1 - self.discount_factor))


    def choose_next_move(self, step=0):
        state = self.determine_state()
        epsilon = np.exp(-1 * self.beta * step)
        self.eps.append(epsilon)
        if random.uniform(0, 1) < epsilon:
            price = random.randint(0, self.num_actions - 1)
        else:
            price = np.argmax(self.q[:, state])
        self.a = price

    def update_rule(self, reward):
        self.q[self.a][self.s] += self.learning_rate * \
            (reward + self.discount_factor *
             np.max(self.q[:, self.s_prime]) - self.q[self.a][self.s])

    def transition(self, my_reward, step):
        self.s_prime = self.determine_state()
        self.update_rule(my_reward)
        self.choose_next_move(step)
        self.s = self.s_prime


if __name__ == "__main__":
    NUM_POSSIBLE_ACTIONS = 15
    
    # TODO: FILL OUT WITH NUMBER OF STATES IN YOUR STATE REPRESENTATION 
    NUM_POSSIBLE_STATES = NUM_POSSIBLE_ACTIONS * NUM_POSSIBLE_ACTIONS
    ####################################################################

    # START SIMULATING THE GAME
    agent1 = CollusionLearner(NUM_POSSIBLE_ACTIONS, NUM_POSSIBLE_STATES)
    agent2 = CollusionLearner(NUM_POSSIBLE_ACTIONS, NUM_POSSIBLE_STATES)
    agents = [agent1, agent2]
    env = CollusionEnv(agents, NUM_POSSIBLE_ACTIONS, NUM_POSSIBLE_STATES)
    env.run(save_figs=True) 
    
    
    
    
