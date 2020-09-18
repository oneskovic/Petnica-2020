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

vector<vector<double>> GaAgent::train(unordered_map<string, double> eval_game_params, int no_threads)
{
	int max_degree = hparams["max_parameter_degree"];
	int no_blues = hparams["no_blue_organisms"];
	int no_reds = hparams["no_red_organisms"];
	int no_parameters = hparams["no_parameters"];
	int coef_count = pow(max_degree + 1, no_parameters);
	auto ga_util = GaUtil(&this->random_util, coef_count);

	vector<double> returns;
	vector<vector<double>> best_genomes;

	int class_count = 100;
	int best_classes = 10;
	TSDeque<training_class> training_classes;
	TSDeque<training_class> evaluated_classes;
	for (size_t i = 0; i < class_count; i++)
	{
		auto blue_genomes = random_util.rand_matrix_double(no_blues, coef_count, -10, 10);
		auto red_genomes = random_util.rand_matrix_double(no_reds, coef_count, -10, 10);
		training_class tc = {
			red_genomes,
			blue_genomes,
			-1,
			GameEnv(blue_genomes, red_genomes, max_degree, 
			&random_util, hparams["food_count"], hparams["board_size"])
		};
		training_classes.push_back(tc);
	}
	for (int gen_number = 0; gen_number < hparams["no_generations"]; gen_number++)
	{
		vector<thread> worker_threads(no_threads);
		for (size_t i = 0; i < no_threads; i++)
		{
			worker_threads[i] = thread(&GaAgent::evaluate_training_classes, this, ref(training_classes), ref(evaluated_classes), 10);
		}
		for (size_t i = 0; i < no_threads; i++)
		{
			worker_threads[i].join();
			worker_threads[i].~thread();
		}

		auto evaluated = evaluated_classes.to_vector();
		sort(evaluated.begin(), evaluated.end());
		evaluated_classes.clear();

		auto max_score = max_element(evaluated.begin(), evaluated.end(),
			[](const training_class& lhs, const training_class& rhs)
			{
				return lhs.score < rhs.score;
			})->score;

		cout << max_score << "\n";
		returns.push_back(max_score);

		for (size_t i = 0; i < class_count; i++)
		{
			int pos1 = random_util.rand_int(best_classes - 1);
			int pos2 = random_util.rand_int(best_classes - 1);
			while (pos2 == pos1)
			{
				pos2 = random_util.rand_int(best_classes - 1);
			}
			training_classes.push_back(combine_classes(evaluated[pos1], evaluated[pos2]));
		}
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

void GaAgent::evaluate_training_classes(TSDeque<training_class>& training_classes, TSDeque<training_class>& evaluated_classes, int no_tests)
{
	while (!training_classes.is_empty())
	{
		training_class tc = training_classes.pop_front();
		double total_return = 0;
		for (int i = 0; i < no_tests; i++)
		{
			auto time_step = tc.eval_env.reset();
			double episode_return = 0;
			while (!time_step.is_last())
			{
				time_step = tc.eval_env.step();
				episode_return += time_step.reward;
			}
			total_return += episode_return;
		}
		tc.score = total_return / no_tests;
		evaluated_classes.push_back(tc);
	}
	
}

GaAgent::training_class GaAgent::combine_classes(training_class& tc1, training_class& tc2)
{
	// Shuffle randomly to ensure fairness
	random_util.random_shuffle(tc1.blue_genomes);
	random_util.random_shuffle(tc1.red_genomes);
	random_util.random_shuffle(tc2.blue_genomes);
	random_util.random_shuffle(tc2.red_genomes);

	// Crossover parents
	int no_genomes = tc1.blue_genomes.size();
	vector<vector<double>> blue_genomes; blue_genomes.reserve(no_genomes);
	blue_genomes.insert(blue_genomes.end(), tc1.blue_genomes.begin(), tc1.blue_genomes.begin() + no_genomes/2);
	blue_genomes.insert(blue_genomes.end(), tc2.blue_genomes.begin(), tc2.blue_genomes.begin() + no_genomes/2);
	
	no_genomes = tc1.red_genomes.size();
	vector<vector<double>> red_genomes; red_genomes.reserve(no_genomes);
	red_genomes.insert(red_genomes.end(), tc1.red_genomes.begin(), tc1.red_genomes.begin() + no_genomes/2);
	red_genomes.insert(red_genomes.end(), tc2.red_genomes.begin(), tc2.red_genomes.begin() + no_genomes/2);

	// Mutate child
	vector<vector<double>> mutation_vec =
		random_util.rand_matrix_double(blue_genomes.size(), blue_genomes[0].size(), -0.2, 0.2);
	for (size_t row = 0; row < blue_genomes.size(); row++)
	{
		for (size_t col = 0; col < blue_genomes[0].size(); col++)
			blue_genomes[row][col] += blue_genomes[row][col] * mutation_vec[row][col];
	}

	mutation_vec = random_util.rand_matrix_double(red_genomes.size(), red_genomes[0].size(), -0.2, 0.2);
	for (size_t row = 0; row < red_genomes.size(); row++)
	{
		for (size_t col = 0; col < red_genomes[0].size(); col++)
			red_genomes[row][col] += red_genomes[row][col] * mutation_vec[row][col];
	}

	training_class combined =
	{
		red_genomes,
		blue_genomes,
		-1,
		GameEnv(blue_genomes, red_genomes,tc1.eval_env.polynomial_degree,
		&random_util,tc1.eval_env.board_food_count,tc1.eval_env.board_length)
	};
	return combined;
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
	vector<vector<double>> organism_function_evaluations; 
	organism_function_evaluations.reserve(blue_organisms.size());
	vector<bool> second_argument = { true, false };
	for (bool arg : second_argument)
	{
		organism_function_evaluations.clear();
		for (size_t i = 0; i < blue_organisms.size(); i++)
		{
			vector<double> x = { 0, (double)arg }; double step = 0.03;
			vector<double> y_values; y_values.reserve(700);
			while (x[0] <= 20)
			{
				double y = blue_organisms[i].compute_function_recursive(&x);
				y_values.push_back(y);
				x[0] += step;
			}
			organism_function_evaluations.push_back(y_values);
		}
		logger->log_to_file(organism_function_evaluations, "functions-blue-"+to_string(arg)+".csv");
	}
	
	organism_function_evaluations.clear();
	organism_function_evaluations.reserve(red_organisms.size());
	for (bool arg : second_argument)
	{
		organism_function_evaluations.clear();
		for (size_t i = 0; i < red_organisms.size(); i++)
		{
			vector<double> x = { 0, (double)arg }; double step = 0.03;
			vector<double> y_values; y_values.reserve(700);
			while (x[0] <= 20)
			{
				double y = red_organisms[i].compute_function_recursive(&x);
				y_values.push_back(y);
				x[0] += step;
			}
			organism_function_evaluations.push_back(y_values);
		}
		logger->log_to_file(organism_function_evaluations, "functions-red-" + to_string(arg) + ".csv");
	}
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
