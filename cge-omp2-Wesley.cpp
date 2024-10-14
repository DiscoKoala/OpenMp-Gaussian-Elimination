//-----------------------------------------------------------------------
// Gaussian elimination program : C++ OpenMP Version 2
//-----------------------------------------------------------------------
// Some features:
// + Rowwise Data layout
// + Rowwise Elimination
// Programming by: Gita Alaghband, Lan Vu, Wesley Johnson
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//-----------------------------------------------------------------------
// Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[], int& n, int& numThreads, int& isPrint) {
    bool isOK = true;
    if (argc < 3) {
        cout << "Arguments:<X> <Y> [<Z>]" << endl;
        cout << "X : Matrix size [X x X]" << endl;
        cout << "Y : Number of threads" << endl;
        cout << "Z = 1: print the input/output matrix if X < 10" << endl;
        cout << "Z <> 1 or missing: does not print the input/output matrix" << endl;
        isOK = false;
    } else {
        n = atoi(argv[1]);
        if (n <= 0) {
            cout << "Matrix size must be larger than 0" << endl;
            isOK = false;
        }
        numThreads = atoi(argv[2]);
        if (numThreads <= 0) {
            cout << "Number of threads must be larger than 0" << endl;
            isOK = false;
        }
        if (argc >= 4)
            isPrint = (atoi(argv[3]) == 1 && n <= 9) ? 1 : 0;
        else
            isPrint = 0;
    }
    return isOK;
}

//-----------------------------------------------------------------------
// Initialize the value of matrix a[n x n]
//-----------------------------------------------------------------------
void InitializeMatrix(float** &a, int n) {
    a = new float*[n];
    a[0] = new float[n * n];
    for (int i = 1; i < n; i++)
        a[i] = a[i - 1] + n;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                a[i][j] = (((float)i + 1) * ((float)i + 1)) / (float)2;
            else
                a[i][j] = (((float)i + 1) + ((float)j + 1)) / (float)2;
        }
    }
}

//------------------------------------------------------------------
// Delete matrix a[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float** a, int n) {
    delete[] a[0];
    delete[] a;
}

//------------------------------------------------------------------
// Print matrix
//------------------------------------------------------------------
void PrintMatrix(float** a, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << "Row " << (i + 1) << ":\t";
        for (int j = 0; j < n; j++) {
            printf("%.2f\t", a[i][j]);
        }
        std::cout << endl;
    }
}

//------------------------------------------------------------------
// Compute the Gaussian Elimination for matrix a[n x n]
//------------------------------------------------------------------
bool ComputeGaussianElimination(float **a, int n) {
    float pivot, gmax, pmax, temp;
    int pindmax, gindmax, i, j, k;

    #pragma omp parallel shared(a, gmax, gindmax) firstprivate(n, k) private(pivot, i, j, temp, pmax, pindmax)
    {
        for (k = 0; k < n - 1; k++) {
            
            // Ensure that the global max variable is initialized once.
            #pragma omp single
            {
                gmax = 0.0;
            }
            
            pmax = 0.0;
            pindmax = 0;
            
            
            #pragma omp for schedule(static)
            for (i = k; i < n; i++) {
                temp = abs(a[i][k]);
                if (temp > pmax) {
                    pmax = temp;
                    pindmax = i;
                }
            }

            // Ensure one thread can write to global variable at a time.
            #pragma omp critical
            {
                if (gmax < pmax) {
                    gmax = pmax;
                    gindmax = pindmax;
                }
            }

            // Prevent further progress until all threads reach this point.
            #pragma omp barrier

            // If gmax is zero, matrix is empty
            if (gmax == 0) break;

            // Swap row if necessary
            if (gindmax != k) {
                #pragma omp for schedule(static)
                for (j = k; j < n; j++) {
                    temp = a[gindmax][j];
                    a[gindmax][j] = a[k][j];
                    a[k][j] = temp;
                }
            }

            pivot = -1.0 / a[k][k];

            // Row reduction
            #pragma omp for schedule(static)
            for (i = k + 1; i < n; i++) {
                temp = pivot * a[i][k];
                for (j = k; j < n; j++) {
                    a[i][j] = a[i][j] + temp * a[k][j];
                }
            }
        }
    }
    return true;
}

//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int n, numThreads, isPrintMatrix;
    float **a;
    double runtime;
    bool isOK;
    if (GetUserInput(argc, argv, n, numThreads, isPrintMatrix) == false) return 1;

    omp_set_num_threads(numThreads);

    InitializeMatrix(a, n);

    if (isPrintMatrix) {
        cout << "The input matrix" << endl;
        PrintMatrix(a, n);
    }

    runtime = omp_get_wtime();

    isOK = ComputeGaussianElimination(a, n);

    runtime = omp_get_wtime() - runtime;

    if (isOK == true) {
        if (isPrintMatrix) {
            cout << "Output matrix:" << endl;
            PrintMatrix(a, n);
        }
        cout << "Gaussian Elimination runs in " << setiosflags(ios::fixed) << setprecision(2) << runtime << " seconds\n";
    } else {
        cout << "The matrix is singular" << endl;
    }

    cout << "Matrix multiplication is computed using max of threads = " << omp_get_max_threads() << " threads or cores" << endl;
    cout << "Matrix size = " << n << endl;

    DeleteMatrix(a, n);

    return 0;
}
