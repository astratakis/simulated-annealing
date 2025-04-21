#include <stdlib.h>
#include "annealing.h"
#include <pthread.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>

problem_t initialize_problem(double *h, double *J, uint32 num_nodes, double temperature, double cooling_rate, double min_temperature, uint32 max_iterations)
{
    problem_t problem;
    problem.num_nodes = num_nodes;

    problem.graph.nodes = (uint8 *)malloc(num_nodes * sizeof(uint8));
    problem.graph.next = (uint8 *)malloc(num_nodes * sizeof(uint8));
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

    memcpy(problem.graph.bias, h, num_nodes * sizeof(double));
    memcpy(problem.graph.adj, J, num_nodes * num_nodes * sizeof(double));

    reset(&problem);

    return problem;
}

void free_problem(problem_t *problem)
{
    free(problem->graph.nodes);
    free(problem->graph.next);
    free(problem->graph.bias);
    free(problem->graph.adj);
    free(problem->best_state);
}

void evolve(problem_t *problem, uint32 num_threads)
{
}

void reset(problem_t *problem)
{
    problem->current_iteration = 0;
    problem->current_energy = 0;
    problem->best_energy = 0;
    memset(problem->best_state, 0, problem->num_nodes * sizeof(uint8));
    memset(problem->graph.next, 0, problem->num_nodes * sizeof(uint8));

    unsigned int seed;
    int success = _rdseed32_step(&seed);
    if (!success)
    {
        fprintf(stderr, "RDSEED failed...\n");
        exit(1);
    }

    srand(seed);

    for (unsigned int i = 0; i < problem->num_nodes; i++)
    {
        problem->graph.nodes[i] = rand() % 2;
    }
}
