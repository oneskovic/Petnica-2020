#pragma once
#include <iostream>
#include <mutex>
using namespace std;
class ProgressBar
{
public:
	ProgressBar(size_t max_progress);
	void Progress(size_t progress_size = 1);
	void ShowBar();
private:
	size_t max_progress;
	size_t current_progress;
	mutex print_mutex;
};