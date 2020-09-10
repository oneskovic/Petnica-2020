#pragma once
#include <vector>
#include <random>
#include <map>
#include <string>
#include <algorithm>
#include <iterator>
#include <numeric>
using namespace std;
class RandomUtil
{
public:
	RandomUtil(size_t seed);
	RandomUtil();
	/// <summary>
	/// Returns an array of given size containing uniform random numbers
	//  In range[min_value, max_value)
	/// </summary>
	vector<double> get_random_array(int size, double min_value, double max_value);
	/// <summary>
	/// Returns an array of given size containing uniform random integer numbers
	//  In range[min_value, max_value]
	/// </summary>
	vector<int> get_random_array_discrete(int size, int min_value, int max_value);
	/// <summary>
	/// Returns a matrix of given size containing uniform random numbers
	/// In range [min_value, max_value)
	/// </summary>
	vector<vector<double>> rand_matrix_double(int no_rows, int no_columns, double min_value, double max_value);
	/// <summary>
	/// Returns a random alpha-numeric string of given length
	/// </summary>
	string rand_string(size_t size);
	/// <summary>
	/// Produces a uniform random integer in range [lower_bound,upper_bound]
	/// </summary>
	int rand_int(int min, int max);
	/// <summary>
	/// Produces a uniform random integer in range [0,upper_bound]
	/// </summary>
	int rand_int(int max);
	/// <summary>
	/// Produces a uniform random double in range [lower_bound,upper_bound)
	/// </summary>
	double rand_double(double min, double max);
	/// <summary>
	/// Reorders the given vector randomly
	/// </summary>
	template <class t>
	void random_shuffle(vector<t>& elements);

	/// <summary>
	/// Produces a vector of required length (count) by picking random elements
	/// with given probabilities
	/// </summary>
	template<typename t>
	vector<t> random_choices(const vector<t>& elements, vector<double> probabilities, int count);
private:
	mt19937 mersenne_twister;
	map<pair<int, int>, uniform_int_distribution<int>> integer_distributions;
	map<pair<double, double>, uniform_real_distribution<double>> real_distributions;
	vector<char> char_set =
	{ '0','1','2','3','4',
	'5','6','7','8','9',
	'A','B','C','D','E','F',
	'G','H','I','J','K',
	'L','M','N','O','P',
	'Q','R','S','T','U',
	'V','W','X','Y','Z',
	'a','b','c','d','e','f',
	'g','h','i','j','k',
	'l','m','n','o','p',
	'q','r','s','t','u',
	'v','w','x','y','z'
	};
};

template<class t>
inline void RandomUtil::random_shuffle(vector<t>& elements)
{
	for (int i = elements.size() - 1; i > 0; i--)
	{
		int rand_pos = rand_int(i);
		swap(elements[i], elements[rand_pos]);
	}
}

template<typename t>
vector<t> RandomUtil::random_choices(const vector<t>& elements, vector<double> probabilities, int count)
{
	vector<t> choices; choices.reserve(count);
	// Make sure probabilities are in range [0,1]
	double probs_sum = accumulate(probabilities.begin(), probabilities.end(), 0.0);
	for (size_t i = 0; i < probabilities.size(); i++)
		probabilities[i] /= probs_sum;

	// Pair up probabilities with elements and sort
	vector<pair<double, t>> probs_elements(elements.size());
	for (size_t i = 0; i < elements.size(); i++)
		probs_elements[i] = { probabilities[i],elements[i] };
	sort(probs_elements.begin(), probs_elements.end(),
		[](const pair<double, t>& lhs, const pair<double, t>& rhs) {
			return lhs.first < rhs.first;
		});

	for (size_t i = 0; i < count; i++)
	{
		double rand_value = rand_double(0, 1);
		double current_prob_sum = 0;
		for (size_t j = 0; j < probs_elements.size(); j++)
		{
			current_prob_sum += probs_elements[j].first;
			if (rand_value <= current_prob_sum || j == probs_elements.size() - 1)
			{
				choices.push_back(probs_elements[j].second);
				break;
			}
		}
	}
	return choices;
}
