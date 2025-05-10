#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "annealing.h"
#include <getopt.h>
#include <fstream>

using namespace std;

#define stream (from_file ? in : cin)

void print_help(const char *progname)
{
    std::cout << "Usage: " << progname << " [options]\n\n"
              << "Options:\n"
              << "  -h, --help                   Show this help message and exit\n"
              << "  -f, --file <FILE_NAME>       Read QUBO matrix from file instead of stdin\n"
              << "  -t, --threads <num_threads>  Number of threads to spawn\n"
              << "  -n, --normalize              Normalize the Ising Hamiltonian\n"
              << "  -o, --output <FILE_NAME>     Write output to specified file\n"
              << "  -g, --gpu <blocks> <threads> Run on GPU with given blocks and threads\n"
              << "  -m, --monitor                Monitor the process\n"
              << "  -i, --iter <max_iter>        Maximum number of iterations\n"
              << "  -r, --ratio <cooling_ratio>  Cooling ratio for annealing\n"
              << std::endl;
}

void print_ising(double *h, double *J, unsigned int n)
{
    cout << n << endl;
    for (unsigned int i = 0; i < n; i++)
    {
        cout << h[i] << (i < n - 1 ? " " : "");
    }
    cout << endl;
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << J[i * n + j] << (j < n - 1 ? " " : "");
        }
        cout << endl;
    }
}

int main(int argc, char *argv[])
{
    string input_file;
    string output_file;
    int num_threads = 1;
    bool normalize = false;
    bool monitor = false;
    bool to_file = false;
    bool from_file = false;
    bool gpu = false;
    int gpu_blocks = 0;
    int gpu_threads = 0;
    int max_iter = 1000;
    double cooling_ratio = 0.95;

    static struct option long_opts[] = {
        {"help", no_argument, nullptr, 'h'},
        {"file", required_argument, nullptr, 'f'},
        {"threads", required_argument, nullptr, 't'},
        {"normalize", no_argument, nullptr, 'n'},
        {"output", required_argument, nullptr, 'o'},
        {"gpu", required_argument, nullptr, 'g'},
        {"monitor", no_argument, nullptr, 'm'},
        {"iter", required_argument, nullptr, 'i'},
        {"ratio", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hf:t:no:g:mi:r:", long_opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_help(argv[0]);
            return 0;

        case 'f':
            input_file = optarg;
            from_file = true;
            break;

        case 't':
            num_threads = std::atoi(optarg);
            break;

        case 'n':
            normalize = true;
            break;

        case 'o':
            to_file = true;
            output_file = optarg;
            break;

        case 'g':
            gpu = true;
            gpu_blocks = std::atoi(optarg);
            if (optind < argc && argv[optind][0] != '-')
            {
                gpu_threads = std::atoi(argv[optind++]);
            }
            else
            {
                cerr << "Error: --gpu requires two arguments.\n";
                return 1;
            }
            break;

        case 'm':
            monitor = true;
            break;

        case 'i':
            max_iter = std::atoi(optarg);
            break;

        case 'r':
            cooling_ratio = std::atof(optarg);
            break;

        case '?':
        default:
            print_help(argv[0]);
            return 1;
        }
    }

    ifstream in;

    if (from_file)
    {
        in.open(input_file);
        if (!in)
        {
            cerr << "Error: could not open " << input_file << endl;
            return 1;
        }
    }

    unsigned int n;
    stream >> n;

    double *Q = (double *)malloc(n * n * sizeof(double));

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            stream >> Q[i * n + j];
        }
    }

    double *J = (double *)malloc(n * n * sizeof(double));
    double *h = (double *)malloc(n * sizeof(double));

    memset(J, 0, n * n * sizeof(double));
    memset(h, 0, n * sizeof(double));

    for (unsigned int i = 0; i < n - 1; i++)
    {
        for (unsigned int j = i + 1; j < n; j++)
        {
            J[i * n + j] = Q[i * n + j] / 4;
        }
    }

    for (unsigned int i = 0; i < n; i++)
    {
        h[i] = Q[i * n + i] / 2;

        for (unsigned int j = i + 1; j < n; j++)
        {
            h[i] += Q[i * n + j] / 4;
        }
    }

    double C = 0.0;

    for (unsigned int i = 0; i < n - 1; i++)
    {
        for (unsigned int j = i + 1; j < n; j++)
        {
            C += Q[i * n + j] / 4;
        }
    }

    for (unsigned int i = 0; i < n; i++)
    {
        C += Q[i * n + i] / 2;
    }

    free(Q);

    if (normalize)
    {
        double max = 0.0;

        for (unsigned int i = 0; i < n; i++)
        {
            for (unsigned int j = 0; j < n; j++)
            {
                max = std::max(max, std::abs(J[i * n + j]));
            }
        }
        for (unsigned int i = 0; i < n; i++)
        {
            max = std::max(max, std::abs(h[i]));
        }

        for (unsigned int i = 0; i < n; i++)
        {
            for (unsigned int j = 0; j < n; j++)
            {
                J[i * n + j] /= max;
            }
        }
        for (unsigned int i = 0; i < n; i++)
        {
            h[i] /= max;
        }
    }

    if (num_threads > n)
    {
        fprintf(stderr, "Cannot spawn more threads than nodes...\n");
        exit(EXIT_FAILURE);
    }

    problem_t problem = initialize_problem(h, J, n, 100.0, cooling_ratio, 1.0, max_iter);

    if (gpu)
    {
        fprintf(stderr, "Not yet implemented...\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        evolve(&problem, num_threads, (monitor ? 1 : 0), C);
    }

    free(h);
    free(J);

    char *string = (char *)malloc(2 * n);

    state(&problem, string, 32);

    cout << string << endl;

    free(string);

    return 0;
}