#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define N sizeof(double)*n

void sub (double * v1, double * v2, double * res, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = v1[i] - v2[i];
    }
}

int main() {
    int n;
    double p;
    std::cin >> n;

    double * v1 = (double *)malloc(N);
    double * v2 = (double *)malloc(N);
    double * res =  (double *)malloc(N);

    for (int i = 0; i < n; i++) {
        std::cin >> p;
        v1[i] = p;
    }
    for (int i = 0; i < n; i++) {
        std::cin >> p;
        v2[i] = p;
    }
    
    sub(v1, v2, res, n);

    for (int i = 0; i < n; i++)
        printf("%lf ", res[i]);
    std::cout << std::endl;

    free(v1);
    free(v2);
    free(res);
    return 0;
}
