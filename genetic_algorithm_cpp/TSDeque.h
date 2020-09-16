#pragma once
#include <vector>
#include <mutex>
#include <deque>
using namespace std;
template <typename t>
class TSDeque
{
public:
	void clear();
	TSDeque();
	void push_back(const t& element);
	t pop_front();
	bool is_empty();
	vector<t> to_vector();
private:
	deque<t> elements;
	mutex mtx;
};

template<typename t>
inline void TSDeque<t>::clear()
{
	mtx.lock();
	elements.clear();
	mtx.unlock();
}

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