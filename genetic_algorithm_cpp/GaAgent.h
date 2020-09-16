#pragma once
#include "GaUtil.h"
#include "Logger.h"
#include "TSDeque.h"
#include "ProgressBar.h"
#include <unordered_map>
#include <string>
#include <chrono>
#include <iostream>
#include <thread>
class GaAgent
{
public:
	GaAgent(unordered_map<string, double> hyperparameters, string log_dir = "", int eval_interval = -1, bool display_moves = false);
	vector<vector<double>> train(unordered_map<string, double> eval_game_params, int no_threads = 4);
	size_t get_random_seed();
	void set_random_seed(size_t seed);
private:
	struct generation
	{
		vector<vector<double>> blue_coefficients;
		vector<vector<double>> red_coefficients;
	};
	struct training_group
	{
		int remaining_trainings;
		generation generation_to_train;
	};
	struct evaluated_organisms
	{
		vector<Organism> blue_organisms;
		vector<Organism> red_organisms;
	};
	Logger* logger;
	unordered_map<string, double> hparams;
	size_t random_seed;
	int eval_interval;
	int exported_episode_count;
	bool display_moves = false;
	bool should_log = false;
	RandomUtil random_util = RandomUtil(0);
	GaUtil ga_util = GaUtil(&random_util,0);

	/// <summary>
	/// Evaluates functions for all organisms for various inputs and logs results to the given file
	/// </summary>
	void evaluate_functions(const vector<Organism>& organisms, string file_name);
	vector<vector<double>> organisms_to_vector(vector<Organism>* organisms);
	void log_functions(vector<Organism>& red_organisms, vector<Organism>& blue_organisms);
	generation train_generation(const generation& previous_generation, const unordered_map<string, double>& hyperparameters, double mutation_stddev);
	/// <summary>
	/// Trains a number of generations starting from the given generation, with given mutation stddev, annealing it down to 0
	/// </summary>
	generation train_generations(const generation& start_generation, const unordered_map<string, double>& hparams, double start_mutation_stddev, int no_generations, ProgressBar& progressbar);
	evaluated_organisms evaluate_generation(const generation& generation, unordered_map<string,double> hparams, int no_evaluations = 10);
public:	
	double evaluate_ga(vector<vector<double>> blue_coeffs, vector<vector<double>> red_coeffs, GameEnv* eval_env, bool display_moves = false, int no_tests = 10);
};

