#include "GaUtil.h"
#include <algorithm>

GaUtil::GaUtil(RandomUtil* rand_util, int coefficients_count)
{
    this->rand_util = rand_util;
    this->coefficients_count = coefficients_count;
}

vector<double> GaUtil::combine_organisms(Organism* organism1, Organism* organism2, vector<double> mutation_factor_range)
{
    auto coefs1 = organism1->coefficients;
    auto coefs2 = organism2->coefficients;
    
    vector<double> child_coefs = vector<double>(coefs1.size());
    for (size_t i = 0; i < coefs1.size(); i++)
        child_coefs[i] = (coefs1[i] + coefs2[i]) / 2.0;
   
    auto mutation_vector = rand_util->get_random_array(coefs1.size(),
        mutation_factor_range[0], mutation_factor_range[1]);
    for (size_t i = 0; i < mutation_vector.size(); i++)
        child_coefs[i] += mutation_vector[i] * child_coefs[i];
    
    return child_coefs;
}

vector<vector<double>> GaUtil::get_coeffs_from_best(vector<Organism>* organisms, int total_count, int no_best, int no_random, vector<double> mutation_factor_range)
{
    sort(organisms->begin(), organisms->end(), [](const auto& lhs, const auto& rhs)
        {
            return lhs.time_alive > rhs.time_alive;
        });
    vector<vector<double>> best_coefs; best_coefs.reserve(total_count);
    for (size_t i = 0; i < total_count-no_random; i++)
    {
        int pos1 = rand_util->rand_int(0, no_best-1);
        int pos2 = rand_util->rand_int(0, no_best-1);
        //Temporary fix
        while (pos2 == pos1)
            pos2 = rand_util->rand_int(0, no_best-1);
        best_coefs.push_back(combine_organisms(&organisms->at(pos1), &organisms->at(pos2), mutation_factor_range));
    }
    for (size_t i = 0; i < no_random; i++)
        best_coefs.push_back(rand_util->get_random_array(coefficients_count, -10, 10));
    
    return best_coefs;
}

bool GaUtil::compare_organisms(Organism organism1, Organism organism2)
{
    return organism1.time_alive > organism2.time_alive;
}
