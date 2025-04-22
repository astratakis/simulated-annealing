#include <stdlib.h>
#include "annealing.h"
#include <pthread.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <assert.h>

typedef struct
{
    int thread_id;
    problem_t *problem;
    unsigned int from;
    unsigned int to;
} thread_args_t;

problem_t initialize_problem(double *h, double *J, uint32 num_nodes, double temperature, double cooling_rate, double min_temperature, uint32 max_iterations)
{
    problem_t problem;
    problem.num_nodes = num_nodes;

    problem.graph.nodes = (uint8 *)malloc(num_nodes * sizeof(uint8));
    problem.graph.next = (uint8 *)malloc(num_nodes * sizeof(uint8));

    problem.temperature = temperature;
    problem.cooling_rate = cooling_rate;
    problem.min_temperature = min_temperature;
    problem.max_iterations = max_iterations;
    problem.current_iteration = 0;
    problem.current_energy = 0;
    problem.best_energy = 0;
    problem.best_state = (uint8 *)malloc(num_nodes * sizeof(uint8));

    problem.graph.adj = J;
    problem.graph.bias = h;

    problem.energy_history = (double *)malloc(max_iterations * sizeof(double));

    reset(&problem);

    return problem;
}

void free_problem(problem_t *problem)
{
    free(problem->graph.nodes);
    free(problem->graph.next);
    free(problem->best_state);
    free(problem->energy_history);
}

void *worker(void *arg)
{
    thread_args_t *t = (thread_args_t *)arg;

    unsigned int from = t->from;
    unsigned int to = t->to;

    for (unsigned int i = from; i < to; i++)
    {
    }

    return NULL;
}

void *step(void *arg)
{
    thread_args_t *t = (thread_args_t *)arg;

    unsigned int from = t->from;
    unsigned int to = t->to;

    for (unsigned int i = from; i < to; i++)
    {
        t->problem->graph.nodes[i] = t->problem->graph.next[i];
    }

    return NULL;
}

void evolve(problem_t *problem, uint32 num_threads, uint8 monitor, double C)
{
    if (num_threads > problem->num_nodes)
    {
        fprintf(stderr, "Cannot spawn more threads than nodes...\n");
        exit(EXIT_FAILURE);
    }

    pthread_t threads[num_threads];
    thread_args_t args[num_threads];
    int rc;

    for (unsigned int i = 0; i < num_threads; i++)
    {
        args[i].thread_id = i;
        args[i].problem = problem;
        args[i].from = i * (problem->num_nodes / num_threads);
        args[i].to = (i + 1) * (problem->num_nodes / num_threads);
    }

    for (unsigned int i = 0; i < problem->max_iterations; i++)
    {
        if (monitor == 1)
        {
            fprintf(stdout, "\r");
            fprintf(stdout, "Running iteration %u/%u ...", i + 1, problem->max_iterations);
        }
        for (unsigned int i = 0; i < num_threads; i++)
        {
            rc = pthread_create(&threads[i],
                                NULL,
                                worker,
                                &args[i]);
            if (rc != 0)
            {
                fprintf(stderr, "pthread_create() failed for thread %d (code %d)\n", i, rc);
                exit(1);
            }
        }

        for (unsigned int i = 0; i < num_threads; i++)
        {
            pthread_join(threads[i], NULL);
        }

        for (unsigned int i = 0; i < num_threads; i++)
        {
            rc = pthread_create(&threads[i],
                                NULL,
                                step,
                                &args[i]);
            if (rc != 0)
            {
                fprintf(stderr, "pthread_create() failed for thread %d (code %d)\n", i, rc);
                exit(1);
            }
        }

        for (unsigned int i = 0; i < num_threads; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    if (monitor)
    {
        fprintf(stdout, "\n");
    }
}

void reset(problem_t *problem)
{
    problem->current_iteration = 0;
    problem->current_energy = 0;
    problem->best_energy = 0;
    memset(problem->best_state, 0, problem->num_nodes * sizeof(uint8));
    memset(problem->graph.next, 0, problem->num_nodes * sizeof(uint8));
    memset(problem->energy_history, 0, problem->max_iterations * sizeof(double));

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

void state(problem_t *problem, char *string)
{
    memset(string, 0, problem->num_nodes);

    for (unsigned int i = 0; i < problem->num_nodes; i++)
    {
        string[i] = 0x30 + problem->graph.nodes[i];
    }
}