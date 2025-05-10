#include <stdlib.h>
#include "annealing.h"
#include <pthread.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <assert.h>

typedef struct
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} barrier_t;

typedef struct
{
    int thread_id;
    problem_t *problem;
    barrier_t *barrier;
    unsigned int from;
    unsigned int to;
} thread_args_t;

void barrier_init(barrier_t *b, int tripCount)
{
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count = 0;
    b->tripCount = tripCount;
}

void barrier_wait(barrier_t *b)
{
    pthread_mutex_lock(&b->mutex);
    b->count++;
    if (b->count == b->tripCount)
    {
        b->count = 0; // reset for reuse
        pthread_cond_broadcast(&b->cond);
    }
    else
    {
        while (pthread_cond_wait(&b->cond, &b->mutex) != 0)
            ;
    }
    pthread_mutex_unlock(&b->mutex);
}

void barrier_destroy(barrier_t *b)
{
    pthread_mutex_destroy(&b->mutex);
    pthread_cond_destroy(&b->cond);
}

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

void compute_chunk(uint8 *curr, uint8 *nxt, unsigned int from, unsigned int to, problem_t *problem)
{
    for (unsigned int s = from; s < to; s++)
    {
        // s represents the current state bit

        double current_energy = 0.0;
        double changed_energy = 0.0;
        for (uint32 node = 0; node < problem->num_nodes; node++)
        {
            if (problem->graph.adj[s * problem->num_nodes + node] == 0 || s == node)
            {
                continue;
            }
            else
            {
                current_energy -= problem->graph.adj[s * problem->num_nodes + node] * (curr[s] == 1 ? 1 : -1);
                changed_energy -= problem->graph.adj[s * problem->num_nodes + node] * (curr[s] == 1 ? -1 : 1);
            }
        }

        current_energy -= problem->graph.bias[s] * (curr[s] == 1 ? 1 : -1);
        changed_energy -= problem->graph.bias[s] * (curr[s] == 1 ? -1 : 1);

        double delta_epsilon = changed_energy - current_energy;

        if (delta_epsilon >= 0)
        {
            // In this case we probabilistically change the state given a temperature
        }
        else
        {
            nxt[s] = (curr[s] == 1 ? 0 : 1);
        }
    }
}

void *worker(void *v_)
{
    thread_args_t *a = (thread_args_t *)v_;
    barrier_t *barrier = a->barrier;
    unsigned int from = a->from;
    unsigned int to = a->to;
    problem_t *problem = a->problem;

    for (uint32 iter = 0; iter < problem->max_iterations; iter++)
    {
        compute_chunk(problem->graph.nodes, problem->graph.next, from, to, problem);

        barrier_wait(barrier);

        if (a->thread_id == 0)
        {
            uint8 *tmp = problem->graph.nodes;
            problem->graph.nodes = problem->graph.next;
            problem->graph.next = tmp;
        }

        barrier_wait(barrier);
    }
    return NULL;
}

void evolve(problem_t *problem, uint32 num_threads, uint8 monitor, double C)
{
    pthread_t threads[num_threads];
    thread_args_t args[num_threads];
    barrier_t barrier;

    barrier_init(&barrier, num_threads);

    for (unsigned int i = 0; i < num_threads; i++)
    {
        args[i].thread_id = i;
        args[i].problem = problem;
        args[i].barrier = &barrier;
        args[i].from = i * (problem->num_nodes / num_threads);
        args[i].to = (i + 1) * (problem->num_nodes / num_threads);
    }

    for (uint32 t = 0; t < num_threads; t++)
    {
        pthread_create(&threads[t], NULL, worker, &args[t]);
    }

    for (uint32 t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    barrier_destroy(&barrier);
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

void state(problem_t *problem, char *string, const unsigned int split_pos)
{
    memset(string, 0, 2 * problem->num_nodes);

    unsigned int index = 0;

    for (unsigned int i = 0; i < problem->num_nodes; i++)
    {
        if (split_pos != 0 && i != 0 && (i % split_pos) == 0)
        {
            string[index] = '\n';
            index++;
        }
        string[index] = 0x30 + problem->graph.nodes[i];
        index++;
    }
}