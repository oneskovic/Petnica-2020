#include "GaUtil.h"
#include <algorithm>

GaUtil::GaUtil(RandomUtil* rand_util, int coefficients_count)
{
    this->rand_util = rand_util;
    this->coefficients_count = coefficients_count;
}

vector<double> GaUtil::combine_organisms(Organism* organism1, Organism* organism2, double mutation_stddev)
{
    auto coefs1 = organism1->coefficients;
    auto coefs2 = organism2->coefficients;
    
    vector<double> child_coefs = vector<double>(coefs1.size());
    for (size_t i = 0; i < coefs1.size(); i++)
        child_coefs[i] = (coefs1[i] + coefs2[i]) / 2.0;
   
    vector<double> mutation_vector(coefs1.size());
    mutation_vector = rand_util->get_random_array(coefs1.size(),
            0, mutation_stddev, "normal");
    
    for (size_t i = 0; i < mutation_vector.size(); i++)
        child_coefs[i] += mutation_vector[i] * child_coefs[i];
    
    return child_coefs;
}

vector<vector<double>> GaUtil::get_coeffs_from_best(vector<Organism>* organisms, int total_count, double mutation_stddev, bool as_parents)
{
    vector<vector<double>> best_coefs; best_coefs.reserve(total_count);
    vector<int> positions(organisms->size());
    vector<double> scores(organisms->size());
    for (size_t i = 0; i < organisms->size(); i++)
    {
        positions[i] = i;
        scores[i] = max(0.0,organisms->at(i).time_alive-20);
    }

    if (as_parents)
    {
        auto parent_pairs = rand_util->random_choices(positions, scores, 2 * total_count);
        for (size_t i = 0; i < 2 * total_count; i += 2)
        {
            int pos1 = parent_pairs[i];
            int pos2 = parent_pairs[i + 1];
            best_coefs.push_back(combine_organisms(&organisms->at(pos1), &organisms->at(pos2), mutation_stddev));
        }
    }
    else
    {
        auto chosen_organisms = rand_util->random_choices(positions, scores, total_count);
        for (size_t i = 0; i < total_count; i++)
        {
            int pos = chosen_organisms[i];
            best_coefs.push_back(organisms->at(pos).coefficients);
        }
    }
    
    return best_coefs;
}

bool GaUtil::compare_organisms(Organism organism1, Organism organism2)
{
    return organism1.time_alive > organism2.time_alive;
}
