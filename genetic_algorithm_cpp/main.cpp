#include "GaAgent.h"
#include "Logger.h"
#include <iostream>
int main()
{
	unordered_map<string, double> hparams =
	{
		{"max_parameter_degree",3},
		{"no_parameters",1},
		{"no_blue_organisms",30},
		{"no_red_organisms",30},
		{"food_count",20},
		{"board_size",20},
		{"no_random_start",20},
		{"no_random_final",0},
		{"no_random_anneal_time",1000},
		{"mutation_factor_min_start",-0.7},
		{"mutation_factor_min_final",0},
		{"mutation_factor_max_start",0.7},
		{"mutation_factor_max_final",0},
		{"mutation_factor_anneal_time",2000},
		{"no_best",5},
		{"no_generations",2000}
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
	auto ga_agent = GaAgent(hparams, log_dir, 10, false);
	//ga_agent.set_random_seed(1428321818);
	vector<vector<double>> blue_genomes = { {-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802},
{-2.12281,1.0442,-2.59919,-1.67802} };
	vector<vector<double>> red_genomes = { {-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922},
{-0.483442,0.79456,2.1166,-0.844922} };
	/*GameEnv eval_env = GameEnv(blue_genomes, red_genomes, 3, &rand_util, 10, 10);
	double v = ga_agent.evaluate_ga(blue_genomes, red_genomes, &eval_env, false, 30);
	averages.push_back(v);*/
	

	auto best_genomes = ga_agent.train(game_params);
	hparams["random_seed"] = ga_agent.get_random_seed();
	logger.log_to_file(best_genomes, "best_genomes.txt");
	logger.log_to_file(hparams, "hyperparameters.json");
	logger.log_to_file(game_params, "gameparams.json");
	return 0;
}