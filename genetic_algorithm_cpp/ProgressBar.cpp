#include "ProgressBar.h"

ProgressBar::ProgressBar(size_t max_progress)
{
	ios::sync_with_stdio(false);
	this->max_progress = max_progress;
	current_progress = 0;
}

void ProgressBar::Progress(size_t progress_size)
{
	current_progress += progress_size;
}

void ProgressBar::ShowBar()
{
	print_mutex.lock();	
	cout << "\r";
	cout << "8";
	double percent_progress = current_progress * 100.0 / max_progress;
	int rounded_progress = round(percent_progress);
	for (size_t i = 0; i < rounded_progress; i++)
	{
		cout << "=";
	}
	cout << "D";
	for (size_t i = 0; i < (100-rounded_progress); i++)
	{
		cout << " ";
	}
	cout << "]";
	cout << " " << percent_progress << "%";
	for (size_t i = 0; i < 10; i++)
		cout << " ";
	print_mutex.unlock();
}
