#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "annealing.h"

using namespace std;

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

int main()
{
    unsigned int n;
    cin >> n;

    double *Q = (double *)malloc(n * n * sizeof(double));

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cin >> Q[i * n + j];
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

    print_ising(h, J, n);

    free(h);
    free(J);

    return 0;
}