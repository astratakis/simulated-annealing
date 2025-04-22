#ifndef ANNEALING
#define ANNEALING

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef unsigned char uint8;
    typedef unsigned int uint32;

    typedef struct mesh
    {
        uint8 *nodes;
        uint8 *next;
        double *bias;
        double *adj;
    } mesh;

    typedef struct problem_t
    {
        uint32 num_nodes;
        mesh graph;
        double temperature;
        double cooling_rate;
        double min_temperature;
        uint32 max_iterations;
        uint32 current_iteration;
        double current_energy;
        double best_energy;
        uint8 *best_state;
        double *energy_history;
    } problem_t;

    problem_t initialize_problem(double *h, double *J, uint32 num_nodes, double temperature, double cooling_rate, double min_temperature, uint32 max_iterations);

    void state(problem_t *problem, char *string);

    void free_problem(problem_t *problem);

    void evolve(problem_t *problem, uint32 num_threads, uint8 monitor, double C);

    void reset(problem_t *problem);

#ifdef __cplusplus
}
#endif

#endif
