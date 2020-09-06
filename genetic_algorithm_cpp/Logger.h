#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include "RandomUtil.h"
#include <filesystem>
#include <unordered_map>
#include <iomanip>
class Logger
{
private:
	RandomUtil rand_util = RandomUtil(0);
	string base_path;
public:
	Logger(string base_path);
	string get_base_path();
	void change_base_path(string new_path);
	template <typename t> void log_to_file(vector<t> data, string file_name, string separator = ",");
	template <typename t> void log_to_file(vector<vector<t>> data, string file_name, string element_separator = ",", string row_separator = "\n");
	/// <summary>
	/// Logs in .json format
	/// </summary>
	template<typename t1, typename t2> void log_to_file(unordered_map<t1, t2> map, string file_name, string separator = ",\n");
};

template<typename t>
inline void Logger::log_to_file(vector<t> data, string file_name, string separator)
{
	ofstream output_stream(base_path + "/" + file_name);
	for (size_t i = 0; i < data.size(); i++)
	{
		output_stream << data[i];
		if (i < data.size() - 1)
			output_stream << separator;
	}
	output_stream.close();
}

template<typename t>
inline void Logger::log_to_file(vector<vector<t>> data, string file_name, string element_separator, string row_separator)
{
	ofstream output_stream(base_path + "/" + file_name);
	for (size_t row = 0; row < data.size(); row++)
	{
		for (size_t col = 0; col < data[row].size(); col++)
		{
			output_stream << data[row][col];
			if (col < data[row].size() - 1)
				output_stream << element_separator;
		}
		if (row < data.size() - 1)
			output_stream << row_separator;
	}
	output_stream.close();
}

template<typename t1, typename t2>
inline void Logger::log_to_file(unordered_map<t1, t2> map, string file_name, string separator)
{
	ofstream output_stream(base_path + "/" + file_name);
	auto kvp = map.begin();
	output_stream << "{\n";
	for (size_t i = 0; i < map.size(); i++)
	{
		output_stream << setprecision(15) << "\t" << "\"" << kvp->first << "\"" << ":" << kvp->second;
		if (i < map.size() - 1)
			output_stream << separator;
		kvp++;
	}
	output_stream << "\n}";
	output_stream.close();
}