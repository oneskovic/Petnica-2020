#include "Logger.h"

Logger::Logger(string base_path)
{
	this->base_path = base_path;
	filesystem::create_directories(base_path); // requires the c++17 standard
}

string Logger::get_base_path()
{
	return base_path;
}

void Logger::change_base_path(string new_path)
{
	this->base_path = new_path;
	filesystem::create_directories(new_path); // requires the c++17 standard

}
