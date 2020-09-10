#include "GaAgent.h"
#include <iostream>

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

vector<vector<double>> GaAgent::train(unordered_map<string, double> eval_game_params)
{
	int max_degree = hparams["max_parameter_degree"];
	int no_blues = hparams["no_blue_organisms"];
	int no_reds = hparams["no_red_organisms"];
	int no_parameters = hparams["no_parameters"];
	int coef_count = pow(max_degree + 1, no_parameters);
	auto ga_util = GaUtil(&this->random_util, coef_count);

	auto blue_coeffs = random_util.rand_matrix_double(no_blues, coef_count, -10, 10);
	auto red_coeffs = random_util.rand_matrix_double(no_reds, coef_count, -10, 10);
	
	vector<double> returns;
	vector<vector<double>> best_genomes;
	double max_avg_score = numeric_limits<double>::min();

	double no_random = hparams["no_random_start"];
	double random_step = (hparams["no_random_final"] - no_random) / hparams["no_random_anneal_time"];
	
	double mutation_factor_min = hparams["mutation_factor_min_start"];
	double mutation_factor_max = hparams["mutation_factor_max_start"];
	double mutation_factor_min_final = hparams["mutation_factor_min_final"];
	double mutation_factor_max_final = hparams["mutation_factor_max_final"];
	double mutation_factor_min_step = (mutation_factor_min_final - mutation_factor_min) / hparams["mutation_factor_anneal_time"];
	double mutation_factor_max_step = (mutation_factor_max_final - mutation_factor_max) / hparams["mutation_factor_anneal_time"];

	GameEnv env = GameEnv(blue_coeffs, red_coeffs, max_degree, &this->random_util, hparams["food_count"], hparams["board_size"]);
	vector<Organism> prev_blue_organisms, prev_red_organisms;
	for (int gen_number = 0; gen_number < hparams["no_generations"]; gen_number++)
	{
		std::cout << "\r Generation: " << gen_number << "                                    ";
		if (gen_number > 0)
		{
			prev_blue_organisms = vector<Organism>(env.dead_blue_organisms.begin(),env.dead_blue_organisms.end());
			prev_red_organisms = vector<Organism>(env.dead_red_organisms.begin(), env.dead_red_organisms.end());
		}
		env = GameEnv(blue_coeffs, red_coeffs, max_degree, &this->random_util, hparams["food_count"], hparams["board_size"]);
		
		// Evaluate the GA
		if (eval_interval > 0 && (gen_number+1)%eval_interval == 0)
		{
			auto eval_blue_coeffs = ga_util.get_coeffs_from_best(&prev_blue_organisms, eval_game_params["no_blue_organisms"], 0, { 0,0 });
			auto eval_red_coeffs = ga_util.get_coeffs_from_best(&prev_red_organisms, eval_game_params["no_red_organisms"], 0, { 0,0 });
			GameEnv eval_env = GameEnv(eval_blue_coeffs, eval_red_coeffs, max_degree, &this->random_util, eval_game_params["food_count"], eval_game_params["board_size"]);

			double avg_return = evaluate_ga(eval_blue_coeffs, eval_red_coeffs, &eval_env, display_moves,30);
			returns.push_back(avg_return);
			int max_time_alive = -1;
			for (auto organism: prev_red_organisms)
			{
				max_time_alive = max(max_time_alive, organism.time_alive);
			}
			cout << "\nGeneration " << gen_number << ":" << avg_return << " max time alive: " << max_time_alive <<"\n";

			if (avg_return > max_avg_score)
			{
				best_genomes.clear();
				best_genomes.reserve(eval_blue_coeffs.size() + eval_red_coeffs.size());
				best_genomes.insert(best_genomes.end(), eval_blue_coeffs.begin(), eval_blue_coeffs.end());
				best_genomes.insert(best_genomes.end(), eval_red_coeffs.begin(), eval_red_coeffs.end());
				max_avg_score = avg_return;
			}
		}
		
		auto time_step = env.reset();
		while (!time_step.is_last())
		{
			time_step = env.step();
		}

		auto blue_organisms = vector<Organism>(env.dead_blue_organisms.begin(), env.dead_blue_organisms.end());
		auto red_organisms = vector<Organism>(env.dead_red_organisms.begin(), env.dead_red_organisms.end());
		vector<double> mutation_factor_range = { mutation_factor_min,mutation_factor_max };
		blue_coeffs = ga_util.get_coeffs_from_best(&blue_organisms, no_blues, round(no_random), mutation_factor_range);
		red_coeffs = ga_util.get_coeffs_from_best(&red_organisms, no_reds,  round(no_random), mutation_factor_range);

		no_random = max(hparams["no_random_final"],no_random+random_step);
		mutation_factor_min = min(mutation_factor_min_final, mutation_factor_min+mutation_factor_min_step);
		mutation_factor_max = max(mutation_factor_max_step, mutation_factor_max+mutation_factor_max_step);
	}

	if (should_log)
	{
		logger->log_to_file(returns, "returns.csv");
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
	vector<vector<double>> organism_function_evaluations;
	vector<int>second_args = { -1,1 };
	for (int second_arg : second_args)
	{
		organism_function_evaluations.clear();
		organism_function_evaluations.reserve(organisms.size());
		for (size_t i = 0; i < organisms.size(); i++)
		{
			vector<double> x = { 0, (double)second_arg }; double step = 0.03;
			vector<double> y_values; y_values.reserve(700);
			while (x[0] <= 20)
			{
				double y = organisms[i].compute_function_recursive(&x);
				y_values.push_back(y);
				x[0] += step;
			}
			organism_function_evaluations.push_back(y_values);
		}
		logger->log_to_file(organism_function_evaluations, file_name + to_string(second_arg) + ".csv");
	}
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
