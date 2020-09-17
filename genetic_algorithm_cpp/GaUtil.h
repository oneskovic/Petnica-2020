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
	vector<double> combine_organisms(Organism* organism1, Organism* organism2, double mutation_stddev);
	/// <summary>
	/// Returns coefficients gotten from combining random pairs of organisms
	/// count - number of coefficients to return
	/// </summary>
	vector<vector<double>> get_coeffs_from_best(vector<Organism>* organisms, int total_count, double mutation_stddev, bool as_parents = true);
	
	template <class T>
	inline void hash_combine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	RandomUtil* rand_util;
private:
	int coefficients_count = 0;
	bool compare_organisms(Organism organism1, Organism organism2);
};

