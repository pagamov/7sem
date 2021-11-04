#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

void LU(double * A, double * L, double * U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            U[i * n + j] = 0;
            L[i * n + j] = 0;
            if (i == j) {
                L[i * n + j] = 1;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i <= j) {
                U[i * n + j] = A[i * n + j];
                
                for (int k = 0; k < i; k++) {
                    U[i * n + j] -= L[i * n + k] * U[k * n + j];
                }
            } 
            else {
                L[i * n + j] = A[i * n + j];
                for (int k = 0; k < j; k++) {
                    L[i * n + j] -= L[i * n + k] * U[k * n + j];
                }
                L[i * n + j] /= U[j + n * j];
            }
        }
    }
}

void prod(double * A, double * L, double * U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                A[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }
}

void sum(double * A, double * L, double * U, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = L[i * n + j] + U[i * n + j];
        }
    }
}

void display(double * M, int n) {
    for (int y = 0; y < n; y++) {
        cout << "\t";
        for (int x = 0; x < n; x++) {
            cout << M[x + y * n] << " ";
        }
        cout << endl;
    }
}

void displayDouble(double * M, int n) {
    for (int y = 0; y < n; y++) {
        // cout << "\t";
        for (int x = 0; x < n; x++) {
            printf("%lf ",M[x + y * n]);
        }
        printf("\n");
    }
}

// __global__ void LUP(double * A, int * SWP, int i, int n, int newidx) {
// 	int pivot = newidx;
//     double piv;
// 	for (int sw = 0; sw < n; sw++) {
//        piv = A[pivot + n * sw];
//        A[pivot + n * sw] = A[i + n * sw];
//        A[i + n * sw] = piv;
//     }
//     for(int j = i+1; j < n; j++) {
//        A[j + n * i] /= A[i + n * i];
//        for(int k = i+1; k < n; k++) 
//            A[j + n * k] -= A[j + n * i] * A[i + n * k];
//     }
// }

void LUP(double * A, double * C, double * P, int n, int * switching) {
    // C = A;
    // double * C = (double *)malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = A[i * n + j];

    //загружаем в матрицу P единичную матрицу
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (i == j) {
                P[i * n + j] = 1;
            } else {
                P[i * n + j] = 0;
            }
        }
            
    
    for( int i = 0; i < n; i++ ) {
        //поиск опорного элемента
        double pivotValue = 0;
        int pivot = -1;
        for(int row = i; row < n; row++) {
            if( abs(C[ row * n + i ]) > pivotValue ) {
                pivotValue = abs(C[ row * n + i ]);
                pivot = row;
            }
        }
        switching[i] = pivot;
        // cout << pivot << " ";
        if( pivotValue != 0 ) {
           //меняем местами i-ю строку и строку с опорным элементом
           double piv;
           for (int sw = 0; sw < n; sw++) {
               piv = P[pivot * n + sw];
               P[pivot * n + sw] = P[i * n + sw];
               P[i * n + sw] = piv;
               
               piv = C[pivot * n + sw];
               C[pivot * n + sw] = C[i * n + sw];
               C[i * n + sw] = piv;
           }
           for( int j = i+1; j < n; j++ ) {
               C[ j * n + i ] /= C[ i * n + i ];
               for( int k = i+1; k < n; k++ ) 
                   C[ j * n + k ] -= C[ j * n + i ] * C[ i * n + k ];
           }
        }
    }
    // cout << endl;
    // free(C);
}

int main() {
    int n;
    cin >> n;
    double * A = (double *)malloc(sizeof(double) * n * n);
    double * L = (double *)malloc(sizeof(double) * n * n);
    double * U = (double *)malloc(sizeof(double) * n * n);
    int * switching = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i * n + j];
        }
    }
    
    // LU(A, L, U, n);
    LUP(A, L, U, n, switching);
    
    // cout << "A" << endl;
    // display(A, n);
    
    // cout << "L" << endl;
    displayDouble(L, n);
    for (int i = 0; i < n; i++) {
        cout << switching[i] << " ";
    }
    cout << endl;
    
    // cout << "U" << endl;
    // display(U, n);
    
    // cout << "Prod" << endl;
    // prod(A, L, U, n);
    // display(A, n);
    
    // cout << "Sum" << endl;
    // sum(A, L, U, n);
    // displayDouble(A, n);
    
    
    free(A);
    free(L);
    free(U);
    free(switching);
    return 0;
}