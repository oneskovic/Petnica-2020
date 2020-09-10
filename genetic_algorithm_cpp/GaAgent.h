#pragma once
#include "GaUtil.h"
#include "Logger.h"
#include <unordered_map>
#include <string>
#include <chrono>
class GaAgent
{
public:
	GaAgent(unordered_map<string, double> hyperparameters, string log_dir = "", int eval_interval = -1, bool display_moves = false);
	vector<vector<double>> train(unordered_map<string, double> eval_game_params);
	size_t get_random_seed();
	void set_random_seed(size_t seed);
private:
	Logger* logger;
	unordered_map<string, double> hparams;
	size_t random_seed;
	int eval_interval;
	int exported_episode_count;
	bool display_moves = false;
	bool should_log = false;
	RandomUtil random_util = RandomUtil(0);
	/// <summary>
	/// Evaluates functions for all organisms for various inputs and logs results to the given file
	/// </summary>
	void evaluate_functions(const vector<Organism>& organisms, string file_name);
	vector<vector<double>> organisms_to_vector(vector<Organism>* organisms);
	void log_functions(vector<Organism>& red_organisms, vector<Organism>& blue_organisms);
public:	
	double evaluate_ga(vector<vector<double>> blue_coeffs, vector<vector<double>> red_coeffs, GameEnv* eval_env, bool display_moves = false, int no_tests = 10);
};

