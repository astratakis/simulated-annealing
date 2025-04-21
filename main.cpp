#include <iostream>
#include <stdlib.h>

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
    return 0;
}