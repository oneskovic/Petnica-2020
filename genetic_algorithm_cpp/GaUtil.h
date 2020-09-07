#pragma once
#include <mutex>
#include <deque>
#include "RandomUtil.h"
#include "game_environment2D_GA.h"
template<typename t> 
class TSDeque
{
public:
	TSDeque();
	void push_back(const t& element);
	t pop_front();
	bool is_empty();
	vector<t> to_vector();
private:
	deque<t> elements;
	mutex mtx;
};

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

template<typename t>
inline TSDeque<t>::TSDeque()
{
}

template<typename t>
inline void TSDeque<t>::push_back(const t& element)
{
	mtx.lock();
	elements.push_back(element);
	mtx.unlock();
}

template<typename t>
inline t TSDeque<t>::pop_front()
{
	mtx.lock();
	if (elements.empty())
	{
		mtx.unlock();
		throw exception("pop_front() on an empty TSDeque");
	}
	t element = elements.front();
	elements.pop_front();
	mtx.unlock();
	return element;
}

template<typename t>
inline bool TSDeque<t>::is_empty()
{
	bool is_empty = false;
	mtx.lock();
	is_empty = elements.empty();
	mtx.unlock();
	return is_empty;
}

template<typename t>
inline vector<t> TSDeque<t>::to_vector()
{
	vector<t> element_vec; element_vec.reserve(elements.size());
	mtx.lock();
	element_vec.insert(element_vec.end(), elements.begin(), elements.end());
	mtx.unlock();
	return element_vec;
}
