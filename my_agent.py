from q_learning import QLearning
from agt_server.local_games.chicken_arena import ChickenArena
from agt_server.agents.test_agents.chicken.mystery_agent.my_agent import MysteryAgent


NUM_TRAINING_ITERATIONS = 20000
NUM_ITERATIONS_PER_PRINT = 1000


class MyChickenAgent(QLearning):
    def __init__(self, name, num_possible_states, num_possible_actions, initial_state, learning_rate, discount_factor, exploration_rate, training_mode, save_path=None) -> None:
        super().__init__(name, num_possible_states, num_possible_actions, initial_state,
                         learning_rate, discount_factor, exploration_rate, training_mode, save_path)

    def determine_state(self):
        # Determines the next state s_prime given the action histories and reward histories
        my_action_hist = self.get_action_history()
        opp_action_hist = self.get_opp_action_history()
        my_util_hist = self.get_util_history()
        
        # TODO: Fill out this function
        if not my_action_hist or not opp_action_hist:
            return 0
        my_last_action = my_action_hist[-1]
        opp_last_action = opp_action_hist[-1]

        # Let continue = 0; swerve = 1
        if my_last_action == 0 and opp_last_action == 0:    # Both continue
            return 0    # return state_0
        elif my_last_action == 0 and opp_last_action == 1:  # I continue, opp swerve
            return 1    # return state_1
        elif my_last_action == 1 and opp_last_action == 0:  # I swerve, opp continue
            return 2
        elif my_last_action == 1 and opp_last_action == 1:
            return 3
        else:
            return 0
        # raise NotImplementedError


if __name__ == "__main__":
    NUM_TRAINING_ITERATIONS = 20000
    NUM_ITERATIONS_PER_PRINT = 1000
    # this agent only uses ??? states
    NUM_POSSIBLE_STATES = 4  # TODO: UPDATE THE NUMBER OF POSSIBLE STATES
    # chicken has 2 possible actions (CONTINUE and SWERVE).
    NUM_POSSIBLE_ACTIONS = 2

    # Feel Free to change any of these parameters!
    INITIAL_STATE = 0
    LEARNING_RATE = 0.05
    DISCOUNT_FACTOR = 0.90
    EXPLORATION_RATE = 0.05
    
    # TODO: Please fill out each of these parameters: 
    name = "max" # Give your agent a name 
    train = True # Set this to false if you want your agent to be purely exploiting and not exploring after training 
    save_q_table = "q-table.npy" # Set to None to not save the q-table, otherwise save it under the file path that you give it E.g. "qtable.npy" 
                        # If the file already exists then it will initialize the q-table using the saved npy file. 
                        # TODO: PLEASE SAVE IT AS AN ACTUAL FILE like "qtable.npy" SO YOU CAN TEST IT
    join_server = False
    ip = ... 
    port = ...

    # START SIMULATING THE GAME
    agent = MyChickenAgent(name, NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS,
                            INITIAL_STATE, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE, train, save_q_table)
    
    if join_server: 
        agent.connect(ip=ip, port=port)
    else: 
        if train: 
            print("TRAINING PERFORMANCE")
            arena = ChickenArena(
                num_rounds=20000,
                timeout=100,
                players=[
                    agent,
                    MysteryAgent("Bok"),
                    MysteryAgent("Cluck"),
                    MysteryAgent("Buck"),
                    MysteryAgent("Bugawk")
                ]
            )
            arena.run()
        print("TESTING PERFORMANCE")
        agent.set_training_mode(False)
        arena = ChickenArena(
            num_rounds=1000,
            timeout=100,
            players=[
                agent,
                MysteryAgent("Bok"),
                MysteryAgent("Cluck"),
                MysteryAgent("Buck"),
                MysteryAgent("Bugawk")
            ]
        )
        arena.run()
