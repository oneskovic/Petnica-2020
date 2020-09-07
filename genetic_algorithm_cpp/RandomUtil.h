#pragma once
#include <vector>
#include <random>
#include <map>
#include <string>
#include <algorithm>
#include <iterator>
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
	//  In range[min_value, max_value)
	/// </summary>
	vector<int> get_random_array_discrete(int size, int min_value, int max_value);
	/// <summary>
	/// Returns a matrix of given size containing uniform random numbers
	/// In range [min_value, max_value)
	/// </summary>
	vector<vector<double>> rand_matrix_double(int no_rows, int no_columns, int min_value, int max_value);
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
	/// Produces a uniform random double in range [lower_bound,upper_bound]
	/// </summary>
	double rand_double(double min, double max);
	template <class t>
	void random_shuffle(vector<t>& elements);
private:
	mt19937 mersenne_twister;
	map<pair<int, int>,uniform_int_distribution<int>> integer_distributions;
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
	for (int i = elements.size()-1; i > 0; i--)
	{
		int rand_pos = rand_int(i);
		swap(elements[i], elements[rand_pos]);
	}
}
