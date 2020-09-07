#include "RandomUtil.h"

RandomUtil::RandomUtil(size_t seed)
{
	mersenne_twister = mt19937(seed);
}

RandomUtil::RandomUtil()
{
	random_device rand_device;
	mersenne_twister = mt19937(rand_device());
}

vector<double> RandomUtil::get_random_array(int size, double min_value, double max_value)
{
	vector<double> rand_array = vector<double>(size);
	for (size_t i = 0; i < size; i++)
		rand_array[i] = rand_double(min_value, max_value);
	
	return rand_array;
}

vector<int> RandomUtil::get_random_array_discrete(int size, int min_value, int max_value)
{
	vector<int> rand_array;
	for (size_t i = 0; i < size; i++)
		rand_array[i] = rand_int(min_value, max_value);
	return rand_array;
}

vector<vector<double>> RandomUtil::rand_matrix_double(int no_rows, int no_columns, int min_value, int max_value)
{
	vector<vector<double>> rand_matrix = vector<vector<double>>(no_rows);
	for (size_t i = 0; i < no_rows; i++)
		rand_matrix[i] = get_random_array(no_columns, min_value, max_value);
	return rand_matrix;
}

string RandomUtil::rand_string(size_t size)
{
	string rand_string = string(size, ' ');
	for (size_t i = 0; i < size; i++)
		rand_string[i] = char_set[rand_int(0, char_set.size()-1)];
	return rand_string;
}

int RandomUtil::rand_int(int min, int max)
{
	std::pair<int, int> interval = pair<int, int>(min, max);
	if (integer_distributions.find(interval) == integer_distributions.end()) //distribution not found
	{
		//Create new distribution
		integer_distributions[interval] = std::uniform_int_distribution<int>(min, max);
	}
	return integer_distributions[interval](mersenne_twister);
}

int RandomUtil::rand_int(int max)
{
	return rand_int(0,max);
}

double RandomUtil::rand_double(double min, double max)
{
	std::pair<double, double> interval = std::pair<double, double>(min, max);
	if (real_distributions.find(interval) == real_distributions.end()) //distribution not found
	{
		//Create new distribution
		real_distributions[interval] = std::uniform_real_distribution<double>(min, max);
	}
	return real_distributions[interval](mersenne_twister);
}
