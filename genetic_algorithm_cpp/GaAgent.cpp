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
	int parallel_populations = hparams["no_parallel_populations"];
	int generations_per_population = hparams["generations_per_population"];
	int no_iterations = hparams["no_iterations"];
	double mutation_stddev = hparams["mutation_stddev_start"];
	int mutation_stddev_anneal_time = hparams["mutation_stddev_anneal_time"];
	double mutation_stddev_step = -mutation_stddev / mutation_stddev_anneal_time;

	ga_util = GaUtil(&this->random_util, coef_count);

	vector<double> returns;
	vector<double> average_red_scores;
	vector<double> average_blue_scores;
	vector<vector<double>> best_genomes;

	TSDeque<training_group> generations_to_train;
	for (size_t i = 0; i < parallel_populations; i++)
	{
		auto blue_coeffs = random_util.rand_matrix_double(no_blues, coef_count, -10, 10);
		auto red_coeffs = random_util.rand_matrix_double(no_reds, coef_count, -10, 10);
		generation current_gen = { blue_coeffs,red_coeffs };
		generations_to_train.push_back({ generations_per_population, current_gen });
	}

	TSDeque<evaluated_organisms> trained_and_evaluated;
	vector<thread> worker_threads = vector<thread>(no_threads);

	for (int iteration_number = 0; iteration_number < no_iterations; iteration_number++)
	{
		try
		{
			ProgressBar progressbar = ProgressBar(parallel_populations * generations_per_population);
			// Start worker threads
			for (size_t i = 0; i < no_threads; i++)
			{
				worker_threads[i] = thread([&generations_to_train, &trained_and_evaluated, this, mutation_stddev, &progressbar]() {
					while (!generations_to_train.is_empty())
					{
						auto current_gen = generations_to_train.pop_front();
						auto trained_gen = train_generations(current_gen.generation_to_train, hparams, mutation_stddev, current_gen.remaining_trainings, progressbar);
						trained_and_evaluated.push_back(evaluate_generation(trained_gen, hparams));
					}
					});
			}
			// Join all threads
			for (size_t i = 0; i < no_threads; i++)
			{
				worker_threads[i].join();
			}
		}
		catch (exception e)
		{
			cout << e.what();
		}
		vector<pair<evaluated_organisms, double>> generation_total_scores;
		double max_score_red = -1, max_score_blue = -1, max_total_game_score = -1;
		double max_avg_blue_score = -1, max_avg_red_score = -1;

		try {
			for (size_t i = 0; i < parallel_populations; i++)
			{
				auto current_population = trained_and_evaluated.pop_front();
				double blue_score_sum = 0.0, red_score_sum = 0.0;
				int blue_count = current_population.blue_organisms.size();
				for (size_t i = 0; i < blue_count; i++)
				{
					double red_score = current_population.red_organisms[i].time_alive;
					double blue_score = current_population.blue_organisms[i].time_alive;
					blue_score_sum += blue_score;
					red_score_sum += red_score;
					max_score_blue = max(max_score_blue, blue_score);
					max_score_red = max(max_score_red, red_score);
				}

				double total_game_score = blue_score_sum + red_score_sum;
				max_avg_blue_score = max(max_avg_blue_score, blue_score_sum / blue_count);
				max_avg_red_score = max(max_avg_red_score, red_score_sum / blue_count);
				max_total_game_score = max(max_total_game_score, total_game_score);
				returns.push_back(total_game_score);
				generation_total_scores.push_back({ current_population,total_game_score });
			}
		}
		catch (exception e)
		{
			cout << e.what();
		}
		try
		{
			// Start new generations
			vector<int> positions(generation_total_scores.size());
			vector<double> scores(generation_total_scores.size());
			for (size_t i = 0; i < generation_total_scores.size(); i++)
			{
				positions[i] = i;
				scores[i] = generation_total_scores[i].second;
			}

			auto parent_pairs = ga_util.rand_util->random_choices(positions, scores, 2 * parallel_populations);
			for (size_t i = 0; i < 2 * parallel_populations; i += 2)
			{
				int pos1 = parent_pairs[i];
				int pos2 = parent_pairs[i + 1];
				auto new_gen = combine_generations(generation_total_scores[pos1].first, generation_total_scores[pos2].first, mutation_stddev);
				generations_to_train.push_back({ generations_per_population,new_gen });
			}

			// Log stats to screen
			cout << "\nIteration: " << iteration_number << "\n"
				<< " average red score:" << max_avg_red_score << "\n"
				<< " max red score:" << max_score_red << "\n"
				<< " average blue score:" << max_avg_blue_score << "\n"
				<< " max blue score:" << max_score_blue << "\n"
				<< " max total game score:" << max_total_game_score << "\n";

			average_blue_scores.push_back(max_avg_blue_score);
			average_red_scores.push_back(max_avg_red_score);
			returns.push_back(max_total_game_score);
			mutation_stddev += mutation_stddev_step; // Anneal stddev
		}
		catch (exception e)
		{
			cout << e.what();
		}
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

GaAgent::generation GaAgent::combine_generations(generation gen1, generation gen2, double mutation_stddev)
{
	random_util.random_shuffle(gen1.blue_coefficients);
	random_util.random_shuffle(gen1.red_coefficients);
	random_util.random_shuffle(gen2.blue_coefficients);
	random_util.random_shuffle(gen2.red_coefficients);

	int no_genomes = gen1.blue_coefficients.size();
	vector<vector<double>> blue_genomes; blue_genomes.reserve(no_genomes);
	blue_genomes.insert(blue_genomes.end(), gen1.blue_coefficients.begin(), gen1.blue_coefficients.begin() + no_genomes / 2);
	blue_genomes.insert(blue_genomes.end(), gen2.blue_coefficients.begin(), gen2.blue_coefficients.begin() + no_genomes / 2);

	no_genomes = gen1.red_coefficients.size();
	vector<vector<double>> red_genomes; red_genomes.reserve(no_genomes);
	red_genomes.insert(red_genomes.end(), gen1.red_coefficients.begin(), gen1.red_coefficients.begin() + no_genomes / 2);
	red_genomes.insert(red_genomes.end(), gen2.red_coefficients.begin(), gen2.red_coefficients.begin() + no_genomes / 2);
	
	auto mutation_vec = random_util.rand_matrix_double(blue_genomes.size(), blue_genomes[0].size(), 0, mutation_stddev, "normal");
	for (size_t row = 0; row < blue_genomes.size(); row++)
	{
		for (size_t col = 0; col < blue_genomes[0].size(); col++)
			blue_genomes[row][col] += blue_genomes[row][col] * mutation_vec[row][col];
	}
	
	mutation_vec = random_util.rand_matrix_double(red_genomes.size(), red_genomes[0].size(), 0, mutation_stddev, "normal");
	for (size_t row = 0; row < red_genomes.size(); row++)
	{
		for (size_t col = 0; col < red_genomes[0].size(); col++)
			red_genomes[row][col] += red_genomes[row][col] * mutation_vec[row][col];
	}

	generation combined = {
		blue_genomes,red_genomes
	};
	
	return combined;
}

GaAgent::generation GaAgent::combine_generations(evaluated_organisms evaluated1, evaluated_organisms evaluated2, double mutation_stddev)
{
	vector<vector<double>> gen1_reds, gen1_blues, gen2_reds, gen2_blues;
	for (size_t i = 0; i < evaluated1.blue_organisms.size(); i++)
	{
		gen1_reds.push_back(evaluated1.red_organisms[i].coefficients);
		gen1_blues.push_back(evaluated1.blue_organisms[i].coefficients);
	}
	for (size_t i = 0; i < evaluated2.blue_organisms.size(); i++)
	{
		gen2_reds.push_back(evaluated2.red_organisms[i].coefficients);
		gen2_blues.push_back(evaluated2.blue_organisms[i].coefficients);
	}
	generation gen1 = { gen1_blues,gen1_reds }, gen2 = { gen2_blues,gen2_reds };
	return combine_generations(gen1,gen2,mutation_stddev);
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
