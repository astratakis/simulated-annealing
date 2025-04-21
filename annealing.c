#include <stdlib.h>
#include "annealing.h"

problem_t initialize_problem(double *h, double *J, uint32 num_nodes, double temperature, double cooling_rate, double min_temperature, uint32 max_iterations)
{
    problem_t problem;
    problem.num_nodes = num_nodes;

    problem.graph.nodes = (uint8 *)malloc(num_nodes * sizeof(uint8));
    problem.graph.bias = (double *)malloc(num_nodes * sizeof(double));
    problem.graph.adj = (double *)malloc(num_nodes * num_nodes * sizeof(double));

    problem.temperature = temperature;
    problem.cooling_rate = cooling_rate;
    problem.min_temperature = min_temperature;
    problem.max_iterations = max_iterations;
    problem.current_iteration = 0;
    problem.current_energy = 0;
    problem.best_energy = 0;
    problem.best_state = (uint8 *)malloc(num_nodes * sizeof(uint8));

    return problem;
}

void free_problem(problem_t *problem)
{
    free(problem->graph.nodes);
    free(problem->graph.bias);
    free(problem->graph.adj);
    free(problem->best_state);
}
