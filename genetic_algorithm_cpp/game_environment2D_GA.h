#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <deque>
#include <random>
#include <list>
#include <unordered_set>
#include <limits>
#include "RandomUtil.h"

using namespace std;
class TimeStep
{
public:
	string ts_type;
	double reward;
	bool is_last();
	TimeStep(string type, double reward);
};
class Organism
{
public:
	double compute_function_recursive(vector<double>* parameters) const;
	Organism(int x_position, int y_position, int energy, int type, int time_to_reproduce, int polynomial_degree = 0, vector<double> coefficients = vector<double>());
	vector<double> to_vector(bool include_coefs = false);
	int time_alive, x_pos, y_pos, id, type, time_to_reproduce, polynomial_max_degree;
	double energy;
	vector<double> coefficients;
private:
	double multiply_params(vector<double>* parameters, vector<int>* times_used) const;
	double sigmoid(double value) const;
	double eval_function(vector<double>* parameters, vector<int>* times_used, int position, int max_degree,int* available_coef_pos, const vector<double>& coefs) const;

};
class GameEnv
{
public:
	// Vectors containing the currently alive organisms
	vector<Organism> blue_organisms;
	vector<Organism> red_organisms;
	vector<Organism> green_organisms;

	// Lists containing the organisms that have died (used to pick the best organisms)
	list<Organism> dead_red_organisms;
	list<Organism> dead_blue_organisms;

	// Coefficients used when resetting (or starting) the game
	vector<vector<double>> blue_starting_coefs;
	vector<vector<double>> red_starting_coefs;

	int polynomial_degree;
	int start_hp = 20;
	int board_length = 10;
	int food_energy = 10;
	int max_moves = 200;
	int current_move_number = 0;
	int board_food_count = 10;
	int reproduction_cooldown = 3;
	bool episode_ended = false;

	GameEnv(vector<vector<double>> blue_start_coefs, vector<vector<double>> red_start_coefs, int polynomial_degree, RandomUtil* rand_util, int food_count = 10, int board_size = 10);
	TimeStep reset();
	TimeStep step();
private:
	RandomUtil* rand_util;
	void reduce_organism_energy(vector<Organism>* organisms);
	void move_organism(Organism* organism, int action);
	double get_distance(Organism* organism1, Organism* organism2);
	/// <summary>
	/// Finds the optimal action for the given set of organisms.
	/// Returns a pair double - maximal function value, int - the action	
	/// </summary>
	pair<double, int> compute_organism_action(Organism* organism, vector<Organism>* other_organisms);
	/// <summary>Computes and processes actions for organisms, computing function values for each organism in other_organisms</summary>
	void process_actions_for_organisms(vector<Organism>* organisms, vector<vector<Organism>*> other_organisms);
	/// <summary>
	/// Checks if any prey should be consumed, adds energy to the adequate predator organism
	/// Sets energy = 0 for any prey that is consumed - dead prey must be removed afterwards
	/// </summary>
	void consume_prey(vector<Organism>* predator_organisms, vector<Organism>* prey_organisms);
	void reproduce_organisms(vector<Organism>* organisms);
	void move_dead_organisms(vector<Organism>* organisms, list<Organism>* dead_organisms);
};

