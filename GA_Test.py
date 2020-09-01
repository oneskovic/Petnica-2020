from GA_Agent import GA_Agent
hparams = {
    'max_parameter_degree': 10, # Degree of polynomial used for function approximation
    'no_parameters': 1,
    'no_blue_organisms': 30, 
    'no_red_organisms': 30,
    'food_count': 30,
    'board_size': 30,
    'no_random_start': 5, # Number of random organisms inserted into each new generation
    'no_random_final': 0,
    'no_random_anneal_time':160, # Number of generations to anneal to final value
    'mutation_factor_range_start': [-0.1,0.1], # When mutating each coefficient will be multiplied by a random value in this range
    'mutation_factor_range_final': [0,0],
    'mutation_factor_range_anneal_time': 160,
    'no_best': 5, # Number of best organisms chosen for the next generation
    'no_generations': 200
    }
game_params = {
    'no_red_organisms': 5,
    'no_blue_organisms': 3,
    'board_size': 10,
    'food_count': 10
}
ga_agent = GA_Agent(hparams,should_log=True,eval_interval=5,display_moves = True)
ga_agent.train(game_params)
