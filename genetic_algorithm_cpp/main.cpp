#include "GaAgent.h"
#include "Logger.h"
#include <iostream>
#include <chrono>
int main()
{
	unordered_map<string, double> hparams =
	{
		{"max_parameter_degree",4},
		{"no_parameters",3},
		{"no_blue_organisms",10},
		{"no_red_organisms",10},
		{"food_count",10},
		{"board_size",10},
		{"mutation_stddev_start",1},
		{"mutation_stddev_anneal_time",8},
		{"no_iterations",10},
		{"no_parallel_populations",10},
		{"generations_per_population",1}
	};

	unordered_map<string, double> game_params =
	{
		{"no_red_organisms",10},
		{"no_blue_organisms",10},
		{"board_size",10},
		{"food_count",10}
	};
	RandomUtil rand_util = RandomUtil();
	string log_dir = "results/GA-" + rand_util.rand_string(8);
	auto logger = Logger(log_dir);
	auto ga_agent = GaAgent(hparams, log_dir, 20, true);
	//ga_agent.set_random_seed(42);
	vector<vector<double>> blue_genomes = { {1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204},
{1.62615,-3.50673,-0.7431,-5.39204} };
	vector<vector<double>> red_genomes = { {-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722},
{-2.74498,-2.50337,-2.00521,1.58722} };
	/*GameEnv eval_env = GameEnv(blue_genomes, red_genomes, 3, &rand_util, 10, 10);
	double v = ga_agent.evaluate_ga(blue_genomes, red_genomes, &eval_env, true, 100);
	cout << v;*/
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	auto best_genomes = ga_agent.train(game_params);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	auto duration = duration_cast<chrono::milliseconds> (end - begin).count();
	cout << "\nTraining took: " << duration << "ms";
	hparams["random_seed"] = ga_agent.get_random_seed();
	logger.log_to_file(best_genomes, "best_genomes.txt");
	logger.log_to_file(hparams, "hyperparameters.json");
	logger.log_to_file(game_params, "gameparams.json");
	return 0;
}