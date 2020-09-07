#pragma once
#include "GaUtil.h"
#include "RandomUtil.h"
#include "Logger.h"
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
using namespace std;
class GaAgent
{
public:
	GaAgent(unordered_map<string, double> hyperparameters, string log_dir = "", int eval_interval = -1, bool display_moves = false);
	vector<vector<double>> train(unordered_map<string, double> eval_game_params, int no_threads = 4);
	size_t get_random_seed();
	void set_random_seed(size_t seed);
private:
	struct training_class
	{
		vector<vector<double>> red_genomes, blue_genomes;
		double score;
		GameEnv eval_env;
		bool operator <( const training_class& rhs)
		{
			return score > rhs.score;
		}
	};
	void evaluate_training_classes(TSDeque<training_class>& training_classes, TSDeque<training_class>& evaluated_classes, int no_tests=100);
	training_class combine_classes(training_class& tc1, training_class& tc2);
	Logger* logger;
	unordered_map<string, double> hparams;
	size_t random_seed;
	int eval_interval;
	int exported_episode_count;
	bool display_moves = false;
	bool should_log = false;
	RandomUtil random_util = RandomUtil(0);
	vector<vector<double>> organisms_to_vector(vector<Organism>* organisms);
	void log_functions(vector<Organism>& red_organisms, vector<Organism>& blue_organisms);
public:	
	double evaluate_ga(vector<vector<double>> blue_coeffs, vector<vector<double>> red_coeffs, GameEnv* eval_env, bool display_moves = false, int no_tests = 10);
};

