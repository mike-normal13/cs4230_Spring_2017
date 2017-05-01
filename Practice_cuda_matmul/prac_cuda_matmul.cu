#include <stdio.h>
#include <stdlib.h>

//extern int cudaMemcpy();
//extern int cudaFree();

extern __global__ void cudaMatMul(int** C, int** A, int** B);

int main(int argc, char** argv)
{
	int N = 16;

	int* A[N];
	int* B[N];

	// result
	int* C[N];

	// cuda guys
	int* A_c[N];
	int* B_c[N];
	int* C_c[N];

	int i = 0;
	int j = 0;
	int k = 0;

	// malloc individual arrays
	for(i = 0; i < N; i++)
	{
		A[i] = (int*) malloc(N * sizeof(int));
		B[i] = (int*) malloc(N * sizeof(int));
		C[i] = (int*) malloc(N * sizeof(int));

		cudaMalloc((void*) &A_c[i], N * sizeof(int));
		cudaMalloc((void*) &B_c[i], N * sizeof(int));
		cudaMalloc((void*) &C_c[i], N * sizeof(int));
	}

	// init data
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			A[i][j] = i + j;
			B[i][j] = i * j;

			C[i][j] = 0;
		}
	}

	// mat mul
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = 0; k < N; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}

	// free arrays
	for(i = 0; i < N; i++)
	{
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}

	return 0;
}