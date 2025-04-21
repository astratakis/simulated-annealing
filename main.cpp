#include <iostream>
#include <stdlib.h>
#include "annealing.h"

using namespace std;

int main()
{
    unsigned int n;
    cin >> n;

    double *h = (double *)malloc(n * sizeof(double));
    double *J = (double *)malloc(n * n * sizeof(double));

    for (unsigned int i = 0; i < n; i++)
    {
        cin >> h[i];
    }

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cin >> J[i * n + j];
        }
    }

    problem_t problem = initialize_problem(h, J, n, 100.0, 1.0, 1.0, 1000);
    
    char *s = (char *)malloc(n);
    state(&problem, s);

    // In this case we want to output the energy history
    cout << n << endl;
    cout << s << endl;
    cout << problem.current_iteration << endl;
    for (unsigned int i=0; i<problem.current_iteration; i++) {
        cout << problem.energy_history[i] << endl;
    }

    free_problem(&problem);
    free(s);

    return 0;
}