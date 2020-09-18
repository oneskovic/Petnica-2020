#include "game_environment2D_GA.h"

bool TimeStep::is_last()
{
    return ts_type == "termination";
}

TimeStep::TimeStep(string type, double reward)
{
    ts_type = type;
    this->reward = reward;
}

double Organism::compute_function_recursive(vector<double>* parameters) const
{
    auto times_used = vector<int>(parameters->size(), -1);
    size_t coef_index = 0;
    int position = 0;
    double result = 0;
    vector<vector<double>> parameter_powers(parameters->size());
    for (size_t i = 0; i < parameter_powers.size(); i++)
    {
        parameter_powers[i] = vector<double>(polynomial_max_degree + 1);
        double current_pow = 1;
        double param = parameters->at(i);
        for (size_t j = 0; j < parameter_powers[i].size(); j++)
        {
            parameter_powers[i][j] = current_pow;
            current_pow *= param;
        }
    }

    while (position != -1)
    {
        if (position == times_used.size())
        {
            double params_multiplied = 1;
            for (size_t i = 0; i < parameters->size(); i++)
                params_multiplied *= parameter_powers[i][times_used[i]];

            //result += sigmoid(coefficients[coef_index++]*multiply_params(parameters,&times_used));
            result += sigmoid(coefficients[coef_index++] * params_multiplied);
            position--;
        }
        else
        {
            if (times_used[position] == polynomial_max_degree)
                times_used[position--] = -1;

            else
                times_used[position++]++;
        }
    }
    return result;
    /*int coef_pos = 0;
    auto times_used = vector<int>(parameters->size(), 0);
    auto res = eval_function(parameters, &times_used, 0, polynomial_max_degree, &coef_pos, coefficients);
    return res;*/
}

Organism::Organism(int x_position, int y_position, int energy, int type, int time_to_reproduce, int polynomial_degree, vector<double> coefficients)
{
    x_pos = x_position;
    y_pos = y_position;
    this->energy = energy;
    this->type = type;
    this->polynomial_max_degree = polynomial_degree;
    this->coefficients = coefficients;
    this->time_to_reproduce = time_to_reproduce;
    time_alive = 0;
    id = rand();
}

Organism::Organism()
{
    time_alive = 0;
}

vector<double> Organism::to_vector(bool include_coefs)
{
    vector<double> organism_vec = { (double)x_pos,(double)y_pos,energy,(double)time_to_reproduce,(double)type,(double)time_alive };
    if (include_coefs)
    {
        organism_vec.reserve(organism_vec.size() + coefficients.size());
        organism_vec.insert(organism_vec.end(), coefficients.begin(), coefficients.end());
    }
    return organism_vec;
}

double Organism::multiply_params(vector<double>* parameters, vector<int>* times_used) const
{
    double result = 1;
    for (int i = 0; i < parameters->size(); i++)
        result *= pow(parameters->at(i), times_used->at(i));

    return result;
}

double Organism::sigmoid(double value) const
{
    //return value;
    //return 1.0 / (1 + exp(-0.1*value));
    return value / (1 + abs(value));
}

double Organism::eval_function(vector<double>* parameters, vector<int>* times_used, int position, int max_degree, int* available_coef_pos, const vector<double>& coefs) const
{
    /* Each recursive call will compute the sum of all parameter combinations
    where each parameter is used max_degree times.Each call returns the sum of
    all combinations starting with the given prefix - eg. if position = 3
    the prefix is the first 3 parameters */
    if (position >= parameters->size())
    {
        double coef = coefs[*available_coef_pos];
        (*available_coef_pos)++;
        return sigmoid(multiply_params(parameters, times_used) * coef);
    }

    double sub_sum = 0;
    for (size_t i = 0; i < max_degree + 1; i++)
    {
        sub_sum += eval_function(parameters, times_used, position + 1, max_degree, available_coef_pos, coefs);
        times_used->at(position) += 1;
    }
    times_used->at(position) = 0;
    return sub_sum;
}

GameEnv::GameEnv(vector<vector<double>> blue_start_coefs, vector<vector<double>> red_start_coefs, int polynomial_degree, RandomUtil* rand_util, int food_count, int board_size)
{
    episode_ended = false;
    this->polynomial_degree = polynomial_degree;
    this->board_food_count = board_food_count;
    this->board_length = board_size;
    this->rand_util = rand_util;

    // Initialize blue organisms
    blue_organisms.reserve(blue_start_coefs.size() * 4);
    for (auto start_coefs : blue_start_coefs)
    {
        int x_pos = rand_util->rand_int(0, board_size - 1);
        int y_pos = rand_util->rand_int(0, board_size - 1);
        blue_organisms.push_back(Organism(x_pos, y_pos, start_hp, 1, reproduction_cooldown, polynomial_degree, start_coefs));
    }
    blue_starting_coefs = blue_start_coefs;

    // Initialize red organisms
    red_organisms.reserve(red_start_coefs.size() * 4);
    for (auto start_coefs : red_start_coefs)
    {
        int x_pos = rand_util->rand_int(0, board_size - 1);
        int y_pos = rand_util->rand_int(0, board_size - 1);
        red_organisms.push_back(Organism(x_pos, y_pos, start_hp, 2, reproduction_cooldown, polynomial_degree, start_coefs));
    }
    red_starting_coefs = red_start_coefs;

    // Generate green organisms
    green_organisms.reserve(board_food_count);
    for (size_t i = 0; i < board_food_count; i++)
    {
        int x_pos = rand_util->rand_int(0, board_size - 1);
        int y_pos = rand_util->rand_int(0, board_size - 1);
        green_organisms.push_back(Organism(x_pos, y_pos, food_energy, 0, reproduction_cooldown));
    }

    this->current_move_number = 0;
}

TimeStep GameEnv::reset()
{
    episode_ended = false;
    blue_organisms = vector<Organism>();
    red_organisms = vector<Organism>();
    green_organisms = vector<Organism>();
    dead_blue_organisms = list<Organism>();
    dead_red_organisms = list<Organism>();

    // Initialize blue organisms
    blue_organisms.reserve(blue_starting_coefs.size() * 4);
    for (auto start_coefs : blue_starting_coefs)
    {
        int x_pos = rand_util->rand_int(0, this->board_length - 1);
        int y_pos = rand_util->rand_int(0, this->board_length - 1);
        blue_organisms.push_back(Organism(x_pos, y_pos, start_hp, 1, reproduction_cooldown, polynomial_degree, start_coefs));
    }

    // Initialize red organisms
    red_organisms.reserve(red_starting_coefs.size() * 4);
    for (auto start_coefs : red_starting_coefs)
    {
        int x_pos = rand_util->rand_int(0, this->board_length - 1);
        int y_pos = rand_util->rand_int(0, this->board_length - 1);
        red_organisms.push_back(Organism(x_pos, y_pos, start_hp, 2, reproduction_cooldown, polynomial_degree, start_coefs));
    }

    // Generate green organisms
    green_organisms.reserve(board_food_count);
    for (size_t i = 0; i < board_food_count; i++)
    {
        int x_pos = rand_util->rand_int(0, this->board_length - 1);
        int y_pos = rand_util->rand_int(0, this->board_length - 1);
        green_organisms.push_back(Organism(x_pos, y_pos, food_energy, 0, reproduction_cooldown));
    }

    this->current_move_number = 0;
    return TimeStep("restart", 0);
}

void GameEnv::move_dead_organisms(vector<Organism>* organisms, list<Organism>* dead_organisms = nullptr)
{
    vector<Organism> alive_organisms; alive_organisms.reserve(organisms->size());
    for (int i = 0; i < organisms->size(); i++)
    {
        if (organisms->at(i).energy <= 0)
        {
            if (dead_organisms != nullptr)
                dead_organisms->push_back(organisms->at(i));
        }
        else
            alive_organisms.push_back(organisms->at(i));
    }
    *organisms = alive_organisms;
}

TimeStep GameEnv::step()
{
    double reward = 0;
    /* The last action ended the episode. Ignore the current action and start
    a new episode.*/
    if (episode_ended)
        return reset();

    reduce_organism_energy(&blue_organisms);
    reduce_organism_energy(&red_organisms);

    move_dead_organisms(&blue_organisms, &dead_blue_organisms);
    move_dead_organisms(&red_organisms, &dead_red_organisms);

    //Make sure episodes don't go on forever.
    if (current_move_number >= max_moves)
    {
        episode_ended = true;
        dead_blue_organisms.insert(dead_blue_organisms.end(), blue_organisms.begin(), blue_organisms.end());
        dead_red_organisms.insert(dead_red_organisms.end(), red_organisms.begin(), red_organisms.end());
    }
    else if (blue_organisms.size() == 0 || red_organisms.size() == 0)
    {
        episode_ended = true;
        if (blue_organisms.size() == 0)
        {
            /*for (size_t i = 0; i < red_organisms.size(); i++)
            {
                red_organisms[i].time_alive += red_organisms[i].energy;
            }*/
            dead_red_organisms.insert(dead_red_organisms.end(), red_organisms.begin(), red_organisms.end());
        }
        if (red_organisms.size() == 0)
        {
            /*for (size_t i = 0; i < blue_organisms.size(); i++)
            {
                blue_organisms[i].time_alive += blue_organisms[i].energy;
            }*/
            dead_blue_organisms.insert(dead_blue_organisms.end(), blue_organisms.begin(), blue_organisms.end());
        }
    }
    else
        current_move_number++;

    if (!episode_ended)
    {
        process_actions_for_organisms(&blue_organisms, { &green_organisms,&red_organisms,&blue_organisms });

        consume_prey(&blue_organisms, &green_organisms);
        consume_prey(&red_organisms, &blue_organisms);
        move_dead_organisms(&green_organisms);
        move_dead_organisms(&blue_organisms, &dead_blue_organisms);

        process_actions_for_organisms(&red_organisms, { &blue_organisms,&red_organisms });

        reward = blue_organisms.size() + red_organisms.size();

        consume_prey(&red_organisms, &blue_organisms);

        /*Remove dead green and blue organisms. This must be done again
        for green and blue as additional organisms might have been consumed*/
        move_dead_organisms(&blue_organisms, &dead_blue_organisms);

        reproduce_organisms(&blue_organisms);
        reproduce_organisms(&red_organisms);

        // Generate new green organisms
        green_organisms.reserve(board_food_count);
        while (green_organisms.size() != board_food_count)
        {
            int x_pos = rand_util->rand_int(0, this->board_length - 1);
            int y_pos = rand_util->rand_int(0, this->board_length - 1);
            green_organisms.push_back(Organism(x_pos, y_pos, food_energy, 0, reproduction_cooldown));
        }
    }
    if (episode_ended)
        return TimeStep("termination", reward);
    else
        return TimeStep("transition", reward);
}

void GameEnv::reduce_organism_energy(vector<Organism>* organisms)
{
    for (int i = 0; i < organisms->size(); i++)
    {
        organisms->at(i).energy -= 1;
        organisms->at(i).time_to_reproduce = max(0, organisms->at(i).time_to_reproduce - 1);
        organisms->at(i).time_alive += 1;
    }
}

void GameEnv::move_organism(Organism* organism, int action)
{
    // Process organism's action
    switch (action)
    {
    case 0:
        organism->x_pos = (organism->x_pos - 1 + board_length) % board_length;
        break;
    case 1:
        organism->x_pos = (organism->x_pos + 1) % board_length;
        break;
    case 2:
        organism->y_pos = (organism->y_pos + 1) % board_length;
        break;
    case 3:
        organism->y_pos = (organism->y_pos - 1 + board_length) % board_length;
        break;
    default:
        throw exception("action should be in range [0,3].");
    }
}

double GameEnv::get_distance(Organism* organism1, Organism* organism2)
{
    double dx = abs(organism1->x_pos - organism2->x_pos);
    double dy = abs(organism1->y_pos - organism2->y_pos);

    if (dx > board_length / 2)
        dx = board_length - dx;

    if (dy > board_length / 2)
        dy = board_length - dy;

    return abs(dx) + abs(dy);
}

pair<double, int> GameEnv::compute_organism_action(Organism* organism, vector<Organism>* other_organisms)
{
    double max_value = numeric_limits<double>::min();
    if (other_organisms->size() == 0)
        return { max_value, 0 };

    /*Organism* max_value_organism = &other_organisms->at(0);
    int best_action = 0;
    vector<int> possible_actions = { 0,1,2,3 };
    int start_x = organism->x_pos;
    int start_y = organism->y_pos;
    for (int i = 0; i < other_organisms->size(); i++)
    {
        auto other_organism = &other_organisms->at(i);
        if (other_organism->id == organism->id)
            continue;
        double distance = get_distance(organism, other_organism);
        vector<double> parameters = { distance };
        double function_value = organism->compute_function_recursive(&parameters);
        if (function_value > max_value)
        {
            max_value_organism = other_organism;
            max_value = function_value;
        }
    }
    bool is_predator = max_value_organism->type > organism->type;
    double best_distance = get_distance(organism, max_value_organism);
    for (int action: possible_actions)
    {
        move_organism(organism, action);
        double current_distance = get_distance(organism, max_value_organism);
        if (is_predator)
        {
            if (current_distance >= best_distance)
            {
                best_action = action;
                best_distance = current_distance;
            }
        }
        else
        {
            if (current_distance <= best_distance)
            {
                best_action = action;
                best_distance = current_distance;
            }
        }
        organism->x_pos = start_x;
        organism->y_pos = start_y;
    }*/

    int best_action = 0;
    vector<int> possible_actions = { 0,1,2,3 };
    int start_x = organism->x_pos;
    int start_y = organism->y_pos;

    for (int action : possible_actions)
    {
        move_organism(organism, action);
        double current_value = 0;
        for (int i = 0; i < other_organisms->size(); i++)
        {
            auto other_organism = &other_organisms->at(i);
            if (other_organism->id == organism->id)
                continue;

            double distance = get_distance(organism, other_organism);
            double is_predator_relative = 0;
            if (other_organism->type > organism->type)
            {
                is_predator_relative = 1;
            }
            else
                is_predator_relative = -1;
            vector<double> parameters = { distance, is_predator_relative, organism->energy };
            double function_value = organism->compute_function_recursive(&parameters);
            current_value += function_value;
        }
        if (current_value > max_value)
        {
            best_action = action;
            max_value = current_value;
        }
        organism->x_pos = start_x;
        organism->y_pos = start_y;
    }
    return { max_value,best_action };
}

void GameEnv::process_actions_for_organisms(vector<Organism>* organisms, vector<vector<Organism>*> other_organisms)
{
    for (int i = 0; i < organisms->size(); i++)
    {
        double max_score = numeric_limits<double>::min();
        int best_action = 0;
        for (auto other_organisms_vec : other_organisms)
        {
            auto best = compute_organism_action(&organisms->at(i), other_organisms_vec);
            if (best.first > max_score)
            {
                max_score = best.first;
                best_action = best.second;
            }
        }
        move_organism(&organisms->at(i), best_action);
    }
}

void GameEnv::consume_prey(vector<Organism>* predator_organisms, vector<Organism>* prey_organisms)
{
    for (int i = 0; i < prey_organisms->size(); i++)
    {
        auto prey = &prey_organisms->at(i);
        for (int j = 0; j < predator_organisms->size(); j++)
        {
            auto predator = &predator_organisms->at(j);
            if (predator->x_pos == prey->x_pos && predator->y_pos == prey->y_pos)
            {
                predator->energy += prey->energy;
                prey->energy = 0;
                break;
            }
        }
    }
}

void GameEnv::reproduce_organisms(vector<Organism>* organisms)
{
    vector<Organism> organisms_to_add;
    organisms_to_add.reserve(organisms->size() / 2);
    for (int i = 0; i < organisms->size(); i++)
    {
        for (int j = 0; j < organisms->size(); j++)
        {
            if (i != j)
            {
                if (organisms->at(i).x_pos == organisms->at(j).x_pos &&
                    organisms->at(i).y_pos == organisms->at(j).y_pos &&
                    organisms->at(i).time_to_reproduce <= 0 &&
                    organisms->at(j).time_to_reproduce <= 0)
                {
                    organisms->at(i).time_to_reproduce = reproduction_cooldown;
                    organisms->at(j).time_to_reproduce = reproduction_cooldown;
                    double child_energy = (organisms->at(i).energy + organisms->at(j).energy) / 2.0;
                    organisms->at(i).energy /= 2;
                    organisms->at(j).energy /= 2;
                    int coef_count = organisms->at(i).coefficients.size();
                    auto child_coefs = vector<double>(coef_count, 0);
                    for (int coef_index = 0; coef_index < coef_count; coef_index++)
                    {
                        child_coefs[coef_index] = (organisms->at(i).coefficients[coef_index] +
                            organisms->at(j).coefficients[coef_index]) / 2.0;
                    }
                    auto child = Organism(organisms->at(i).x_pos, organisms->at(i).y_pos, child_energy,
                        organisms->at(i).type, reproduction_cooldown, organisms->at(i).polynomial_max_degree,
                        child_coefs);
                    organisms_to_add.push_back(child);
                }
            }
        }
    }
    organisms->reserve(organisms->size() + organisms_to_add.size());
    organisms->insert(organisms->end(), organisms_to_add.begin(), organisms_to_add.end());
}
