#include <stdio.h>
#include <stdlib.h>

//extern int cudaMemcpy();
//extern int cudaFree();

extern __global__ 
void cudaMatMul(int** C, int** A, int** B);

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

	// cuda result placed in this value
	int* ret[N];

	int i = 0;
	int j = 0;

	// malloc individual arrays
	for(i = 0; i < N; i++)
	{
		A[i] = (int*) malloc(N * sizeof(int));
		B[i] = (int*) malloc(N * sizeof(int));
		C[i] = (int*) malloc(N * sizeof(int));

		cudaMalloc((void**) &A_c[i], N * sizeof(int));
		cudaMalloc((void**) &B_c[i], N * sizeof(int));
		cudaMalloc((void**) &C_c[i], N * sizeof(int));

		cudaMalloc((void**) &ret[i], N * sizeof(int));
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

	// COPY TO device memory
	for(i = 0; i < N; i++)
	{
		cudaMemcpy(A_c[i], A[i], N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(B_c[i], B[i], N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(C_c[i], C[i], N * sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMatMul(C_c, A_c, B_c);	

	for(i = 0; i < N; i++)
	{
		cudaMemcpy(ret[i], C_c[i], N * sizeof(int), cudaMemcpyDeviceToHost);
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			printf("%d ", ret[i][j]);
		printf("\n");
	}
	

	// free arrays
	for(i = 0; i < N; i++)
	{
		free(A[i]);
		free(B[i]);
		free(C[i]);

		cudaFree(A_c);
		cudaFree(B_c);
		cudaFree(C_c);

		cudaFree(ret);
	}

	return 0;
}

extern __global__ 
void cudaMatMul(int** C, int** A, int** B)
{
	int i, j, k = 0;	

	// mat mul
	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++)
			for(k = 0; k < N; k++)
				C[i][j] += A[i][k] * B[k][j];
}

