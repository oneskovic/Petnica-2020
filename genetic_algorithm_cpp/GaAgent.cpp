#include "GaAgent.h"

GaAgent::GaAgent(unordered_map<string, double> hyperparameters, string log_dir, int eval_interval, bool display_moves)
{
	this->hparams = hyperparameters;
	random_device rand_device;
	this->random_seed = rand_device();
	if (log_dir != "")
		should_log = true;
	else
		should_log = false;
	this->display_moves = display_moves;
	this->eval_interval = eval_interval;
	this->random_util = RandomUtil(random_seed);
	if (should_log)
		this->logger = new Logger(log_dir);
}

vector<vector<double>> GaAgent::train(unordered_map<string, double> eval_game_params, int no_threads)
{
	int max_degree = hparams["max_parameter_degree"];
	int no_blues = hparams["no_blue_organisms"];
	int no_reds = hparams["no_red_organisms"];
	int no_parameters = hparams["no_parameters"];
	int coef_count = pow(max_degree + 1, no_parameters);
	ga_util = GaUtil(&this->random_util, coef_count);

	vector<double> returns;
	vector<double> average_red_scores;
	vector<double> average_blue_scores;
	vector<vector<double>> best_genomes;
	double max_avg_score = numeric_limits<double>::min();

	double mutation_stddev = hparams["mutation_stddev_start"];
	int mutation_stddev_anneal_time = hparams["mutation_stddev_anneal_time"];
	double mutation_stddev_step = - mutation_stddev / mutation_stddev_anneal_time;

	int parallel_generations = 10;
	TSDeque<training_group> generations_to_train;
	for (size_t i = 0; i < parallel_generations; i++)
	{
		auto blue_coeffs = random_util.rand_matrix_double(no_blues, coef_count, -10, 10);
		auto red_coeffs = random_util.rand_matrix_double(no_reds, coef_count, -10, 10);
		generation current_gen = { blue_coeffs,red_coeffs };
		generations_to_train.push_back({ 10, current_gen });
	}

	TSDeque<evaluated_organisms> trained_and_evaluated;
	vector<thread> worker_threads = vector<thread>(no_threads);

	int no_iterations = 50;
	for (int iteration_number = 0; iteration_number < no_iterations; iteration_number++)
	{
		ProgressBar progressbar = ProgressBar(parallel_generations * 10);
		// Start worker threads
		for (size_t i = 0; i < no_threads; i++)
		{
			worker_threads[i] = thread([&generations_to_train, &trained_and_evaluated, this, mutation_stddev,&progressbar]() {
				while (!generations_to_train.is_empty())
				{
					auto current_gen = generations_to_train.pop_front();
					auto trained_gen = train_generations(current_gen.generation_to_train, hparams, mutation_stddev, current_gen.remaining_trainings, progressbar);
					trained_and_evaluated.push_back(evaluate_generation(trained_gen,hparams));
				}
			});
		}
		// Join all threads
		for (size_t i = 0; i < no_threads; i++)
			worker_threads[i].join();
		
		// Collect all trained organisms into a single vector
		vector<Organism> all_red_organisms, all_blue_organisms;
		for (size_t i = 0; i < parallel_generations; i++)
		{
			auto evaluated_organisms = trained_and_evaluated.pop_front();
			all_red_organisms.insert(all_red_organisms.end(), evaluated_organisms.red_organisms.begin(), evaluated_organisms.red_organisms.end());
			all_blue_organisms.insert(all_blue_organisms.end(), evaluated_organisms.blue_organisms.begin(), evaluated_organisms.blue_organisms.end());
		}

		// Collect statistics
		double max_score_red = -1, max_score_blue = -1;
		double red_score_sum = 0, blue_score_sum = 0;
		for (size_t i = 0; i < all_red_organisms.size(); i++)
		{
			double current_score = all_red_organisms[i].time_alive;
			red_score_sum += current_score;
			max_score_red = max(max_score_red, current_score);
		}
		for (size_t i = 0; i < all_blue_organisms.size(); i++)
		{
			double current_score = all_blue_organisms[i].time_alive;
			blue_score_sum += current_score;
			max_score_blue = max(max_score_blue, current_score);
		}
		double max_total_game_score = -1;
		for (size_t game_index = 0; game_index < all_red_organisms.size()/no_reds; game_index++)
		{
			double total_game_score = 0;
			for (size_t i = game_index*no_reds; i < (game_index+1)*no_reds; i++)
			{
				total_game_score += all_red_organisms[i].time_alive;
				total_game_score += all_blue_organisms[i].time_alive;
			}
			max_total_game_score = max(total_game_score, max_total_game_score);
		}

		// Start new generations
		for (size_t i = 0; i < parallel_generations; i++)
		{
			generation new_gen = { ga_util.get_coeffs_from_best(&all_blue_organisms,no_blues,mutation_stddev),
			ga_util.get_coeffs_from_best(&all_red_organisms,no_reds,mutation_stddev,false) };
			generations_to_train.push_back({10, new_gen });
		}
		// Log stats to screen
		cout << "\nIteration: " << iteration_number << "\n"
			<< " average red score:" << red_score_sum / all_red_organisms.size() << "\n"
			<< " max red score:" << max_score_red << "\n"
			<< " average blue score:" << blue_score_sum / all_blue_organisms.size() << "\n"
			<< " max blue score:" << max_score_blue << "\n"
			<< " max total game score:" << max_total_game_score << "\n";

		average_blue_scores.push_back(blue_score_sum / all_blue_organisms.size());
		average_red_scores.push_back(red_score_sum / all_red_organisms.size());
		returns.push_back(max_total_game_score);
		mutation_stddev += mutation_stddev_step; // Anneal stddev
	}

	if (should_log)
	{
		logger->log_to_file(returns, "returns.csv");
		logger->log_to_file(average_blue_scores, "best-blue-scores.csv");
		logger->log_to_file(average_red_scores, "best-red-scores.csv");
	}
	return best_genomes;
}

size_t GaAgent::get_random_seed()
{
	return random_seed;
}

void GaAgent::set_random_seed(size_t seed)
{
	this->random_util = RandomUtil(seed);
	this->random_seed = seed;
	srand(seed);
}

void GaAgent::evaluate_functions(const vector<Organism>& organisms, string file_name)
{
	/*vector<vector<double>> organism_function_evaluations;
	vector<int>second_args = { -1,1 };
	for (int second_arg : second_args)
	{
		organism_function_evaluations.clear();
		organism_function_evaluations.reserve(organisms.size());
		for (size_t i = 0; i < organisms.size(); i++)
		{
			vector<double> y_values;
			int x = 0;
			while (x <= 10)
			{
				vector<double> parameters = { (double)x, (double)second_arg };
				double y = organisms[i].compute_function_recursive(&parameters);
				y_values.push_back(y);
				x++;
			}
			organism_function_evaluations.push_back(y_values);
		}
		logger->log_to_file(organism_function_evaluations, file_name + to_string(second_arg) + ".csv");
	}*/
	vector<vector<double>> organism_coefs;
	for (auto organism: organisms)
		organism_coefs.push_back(organism.coefficients);
	logger->log_to_file(organism_coefs, file_name);

}

vector<vector<double>> GaAgent::organisms_to_vector(vector<Organism>* organisms)
{
	vector<vector<double>> organisms_vec; organisms_vec.reserve(organisms->size());
	for (size_t i = 0; i < organisms->size(); i++)
		organisms_vec.push_back(organisms->at(i).to_vector());
	return organisms_vec;
}

void GaAgent::log_functions(vector<Organism>& red_organisms, vector<Organism>& blue_organisms)
{
	evaluate_functions(red_organisms, "red-organisms");
	evaluate_functions(blue_organisms, "blue-organisms");
}

GaAgent::evaluated_organisms GaAgent::evaluate_generation(const generation& generation, unordered_map<string, double> hparams, int no_evaluations)
{
	int max_degree = hparams.at("max_parameter_degree");
	int food_count = hparams.at("food_count");
	int board_size = hparams.at("board_size");
	auto env = GameEnv(generation.blue_coefficients, generation.red_coefficients,
		max_degree, &this->random_util, food_count, board_size);

	unordered_map<size_t, pair<Organism, int>> organisms_hashed;
	vector<vector<double>> all_coefs = generation.blue_coefficients;
	all_coefs.insert(all_coefs.end(), generation.red_coefficients.begin(), generation.red_coefficients.end());

	// Add all starting organisms to hashed organisms to avoid adding children later
	for (auto coefs : all_coefs)
	{
		size_t organism_hash = 0;
		for (size_t coef_pos = 0; coef_pos < coefs.size(); coef_pos++)
			ga_util.hash_combine(organism_hash, coefs[coef_pos]);
		organisms_hashed[organism_hash] = { Organism(),0 };
	}

	// Evaluate the current generation a couple of times to get an average score
	for (size_t i = 0; i < no_evaluations; i++)
	{
		auto time_step = env.reset();
		while (!time_step.is_last())
		{
			time_step = env.step();
		}

		auto organisms = vector<Organism>(env.dead_blue_organisms.begin(), env.dead_blue_organisms.end());
		organisms.insert(organisms.end(), env.dead_red_organisms.begin(), env.dead_red_organisms.end());

		for (auto organism : organisms)
		{
			size_t organism_hash = 0;
			for (size_t coef_pos = 0; coef_pos < organism.coefficients.size(); coef_pos++)
				ga_util.hash_combine(organism_hash, organism.coefficients[coef_pos]);

			// If organism wasn't already added it is a child organism and should not be picked
			// for the next generation
			if (organisms_hashed.find(organism_hash) != organisms_hashed.end())
			{
				int total_time_alive = organisms_hashed[organism_hash].first.time_alive + organism.time_alive;
				organisms_hashed[organism_hash].first = organism;
				organisms_hashed[organism_hash].first.time_alive = total_time_alive;
				organisms_hashed[organism_hash].second++;
			}
		}
	}

	vector<Organism> final_blue_organisms, final_red_organisms;
	for (auto kvp : organisms_hashed)
	{
		kvp.second.first.time_alive /= kvp.second.second;
		if (kvp.second.first.type == 1)
		{
			final_blue_organisms.push_back(kvp.second.first);
		}
		else if (kvp.second.first.type == 2)
		{
			final_red_organisms.push_back(kvp.second.first);
		}
	}

	return { final_blue_organisms,final_red_organisms };
}

GaAgent::generation GaAgent::train_generation(const generation& previous_generation, const unordered_map<string, double>& hyperparameters, double mutation_stddev)
{
	auto blue_coeffs = previous_generation.blue_coefficients;
	auto red_coeffs = previous_generation.red_coefficients;
	
	auto evaluated = evaluate_generation(previous_generation, hyperparameters);

	int no_blues = hyperparameters.at("no_blue_organisms");
	int no_reds = hyperparameters.at("no_red_organisms");

	blue_coeffs = ga_util.get_coeffs_from_best(&evaluated.blue_organisms, no_blues, mutation_stddev);
	red_coeffs = ga_util.get_coeffs_from_best(&evaluated.red_organisms, no_reds, mutation_stddev);

	return { blue_coeffs,red_coeffs };
}

GaAgent::generation GaAgent::train_generations(const generation& start_generation, const unordered_map<string, double>& hparams, double start_mutation_stddev, int no_generations, ProgressBar& progressbar)
{
	double mutation_dev_step = -start_mutation_stddev / no_generations;
	double stddev = start_mutation_stddev;
	auto current_gen = start_generation;
	for (size_t i = 0; i < no_generations; i++)
	{
		current_gen = train_generation(current_gen, hparams, stddev);
		stddev += mutation_dev_step;
		progressbar.Progress();
		if ((i+1)%5 == 0)
		{
			progressbar.ShowBar();
		}
	}
	return current_gen;
}


double GaAgent::evaluate_ga(vector<vector<double>> blue_coeffs, vector<vector<double>> red_coeffs, GameEnv* eval_env, bool display_moves, int no_tests)
{
	if (display_moves)
	{
		string original_path = logger->get_base_path();
		logger->change_base_path(original_path + "/episodes/" + to_string(exported_episode_count));
		auto time_step = eval_env->reset();
		log_functions(eval_env->red_organisms, eval_env->blue_organisms);
		int episode_count = 0;
		while (!time_step.is_last())
		{
			auto red_organisms = organisms_to_vector(&eval_env->red_organisms);
			auto blue_organisms = organisms_to_vector(&eval_env->blue_organisms);
			auto green_organisms = organisms_to_vector(&eval_env->green_organisms);
			vector<vector<double>> organisms; 
			organisms.reserve(red_organisms.size() + blue_organisms.size() + green_organisms.size());
			organisms.insert(organisms.end(), red_organisms.begin(), red_organisms.end());
			organisms.insert(organisms.end(), blue_organisms.begin(), blue_organisms.end());
			organisms.insert(organisms.end(), green_organisms.begin(), green_organisms.end());
			logger->log_to_file(organisms, to_string(episode_count) + ".csv");
			time_step = eval_env->step();
			episode_count++;
		}
		logger->change_base_path(original_path);
		exported_episode_count++;
	}

	double total_return = 0;
	for (int i = 0; i < no_tests; i++)
	{
		auto time_step = eval_env->reset();
		double episode_return = 0;
		while (!time_step.is_last())
		{
			time_step = eval_env->step();
			episode_return += time_step.reward;
		}
		total_return += episode_return;
	}

	return total_return / no_tests;
}
