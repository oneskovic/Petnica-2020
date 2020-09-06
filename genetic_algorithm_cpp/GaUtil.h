#pragma once
#include "RandomUtil.h"
#include "game_environment2D_GA.h"
class GaUtil
{
public:
	GaUtil(RandomUtil* rand_util, int coefficients_count);
	/// <summary>
	/// Combines coefficients from organism1 and organism2
	/// Mutates the childand returns the coefficients
	/// </summary>
	vector<double> combine_organisms(Organism* organism1, Organism* organism2, vector<double> mutation_factor_range);
	/// <summary>
	/// Returns coefficients gotten from combining random pairs of organisms
	/// count - number of coefficients to return
	/// </summary>
	vector<vector<double>> get_coeffs_from_best(vector<Organism>* organisms, int total_count, int no_best, int no_random, vector<double> mutation_factor_range);
private:
	RandomUtil* rand_util;
	int coefficients_count = 0;
	bool compare_organisms(Organism organism1, Organism organism2);
};

