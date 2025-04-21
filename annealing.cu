#include "annealing.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 1 // Single thread for one SA run; adjust for parallel runs

// Device function to compute energy of a state
__device__ double compute_energy_device(const uint8 *state,
                                        const double *bias,
                                        const double *adj,
                                        uint32 num_nodes)
{
    double E = 0.0;
    // bias term
    for (uint32 i = 0; i < num_nodes; ++i)
    {
        E += bias[i] * (state[i] ? 1.0 : -1.0);
    }
    // pairwise interactions (undirected, assume J_ij stored for all i,j)
    for (uint32 i = 0; i < num_nodes; ++i)
    {
        for (uint32 j = i + 1; j < num_nodes; ++j)
        {
            double Jij = adj[i * num_nodes + j];
            double s_i = state[i] ? 1.0 : -1.0;
            double s_j = state[j] ? 1.0 : -1.0;
            E += Jij * s_i * s_j;
        }
    }
    return E;
}

// Kernel performing simulated annealing (single-threaded here)
__global__ void anneal_kernel(uint8 *d_state,
                              const double *d_bias,
                              const double *d_adj,
                              uint32 num_nodes,
                              double temperature,
                              double cooling_rate,
                              double min_temperature,
                              uint32 max_iterations,
                              double *d_best_energy,
                              uint8 *d_best_state,
                              unsigned long seed)
{
    // Only one thread does the work
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Initialize RNG
    curandState_t rng;
    curand_init(seed, 0, 0, &rng);

    // Random initial state
    for (uint32 i = 0; i < num_nodes; ++i)
    {
        d_state[i] = curand(&rng) & 1;
    }

    double curr_E = compute_energy_device(d_state, d_bias, d_adj, num_nodes);
    double best_E = curr_E;
    // Copy initial best state
    for (uint32 i = 0; i < num_nodes; ++i)
    {
        d_best_state[i] = d_state[i];
    }

    // Main annealing loop
    for (uint32 iter = 0; iter < max_iterations && temperature > min_temperature; ++iter)
    {
        // Propose a random spin flip
        uint32 i = curand(&rng) % num_nodes;
        // Compute change in energy: ΔE = E(new) - E(curr)
        // Flip spin i locally
        uint8 old_si = d_state[i];
        double s_i = old_si ? 1.0 : -1.0;
        double delta = 2.0 * s_i * d_bias[i];
        // interaction term
        for (uint32 j = 0; j < num_nodes; ++j)
        {
            if (j == i)
                continue;
            double Jij = d_adj[i * num_nodes + j];
            double s_j = d_state[j] ? 1.0 : -1.0;
            delta += 2.0 * s_i * Jij * s_j;
        }

        // Metropolis criterion
        if (delta < 0 || curand_uniform_double(&rng) < exp(-delta / temperature))
        {
            // Accept flip
            d_state[i] = !old_si;
            curr_E += delta;
            // Update best
            if (curr_E < best_E)
            {
                best_E = curr_E;
                for (uint32 k = 0; k < num_nodes; ++k)
                {
                    d_best_state[k] = d_state[k];
                }
            }
        }

        // Cool down
        temperature *= cooling_rate;
    }

    // Write result back
    *d_best_energy = best_E;
}

// Host function to launch annealing on GPU
extern "C" void run_annealing(problem_t *problem)
{
    // Allocate device memory
    uint8 *d_state = nullptr;
    double *d_bias = nullptr;
    double *d_adj = nullptr;
    double *d_best_energy = nullptr;
    uint8 *d_best_state = nullptr;

    uint32 n = problem->num_nodes;
    size_t nodes_sz = n * sizeof(uint8);
    size_t bias_sz = n * sizeof(double);
    size_t adj_sz = n * n * sizeof(double);

    cudaMalloc(&d_state, nodes_sz);
    cudaMalloc(&d_bias, bias_sz);
    cudaMalloc(&d_adj, adj_sz);
    cudaMalloc(&d_best_energy, sizeof(double));
    cudaMalloc(&d_best_state, nodes_sz);

    // Copy data to device
    cudaMemcpy(d_bias, problem->graph.bias, bias_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj, problem->graph.adj, adj_sz, cudaMemcpyHostToDevice);

    // Launch kernel
    unsigned long seed = 42UL; // or use time-based seed
    anneal_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_state, d_bias, d_adj,
        n,
        problem->temperature,
        problem->cooling_rate,
        problem->min_temperature,
        problem->max_iterations,
        d_best_energy,
        d_best_state,
        seed);

    cudaDeviceSynchronize();

    // Copy results back to host
    double best_E;
    cudaMemcpy(&best_E, d_best_energy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(problem->best_state, d_best_state, nodes_sz, cudaMemcpyDeviceToHost);

    problem->best_energy = best_E;

    // Free device memory
    cudaFree(d_state);
    cudaFree(d_bias);
    cudaFree(d_adj);
    cudaFree(d_best_state);
    cudaFree(d_best_energy);
}
