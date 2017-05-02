#include <stdio.h>
#include <stdlib.h>

//extern int cudaMemcpy();
//extern int cudaFree();

extern __global__ 
void cudaMatMul(int** C, int** A, int** B, int n);

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
		A[i] = (void*) malloc(N * sizeof(int));
		B[i] = (void*) malloc(N * sizeof(int));
		C[i] = (void*) malloc(N * sizeof(int));

		cudaMalloc((void**) &A_c[i], N * sizeof(int));
		cudaMalloc((void**) &B_c[i], N * sizeof(int));
		cudaMalloc((void**) &C_c[i], N * sizeof(int));

		//cudaMalloc((void**) &ret[i], N * sizeof(int));

		ret[i] = (int*) malloc(N * sizeof(int));
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
		for(j = 0; j < N; j++)
		{
			cudaMemcpy(A_c[i][j], A[i][j], sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(B_c[i][j], B[i][j], sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(C_c[i][j], C[i][j], sizeof(int), cudaMemcpyHostToDevice);
		}
	}

	cudaMatMul<<<1, 1>>>(C_c, A_c, B_c, N);	

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			cudaMemcpy(ret[i][j], C_c[i][j], sizeof(int), cudaMemcpyDeviceToHost);
		}
	}

	printf("segfault before?\n");
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			printf("%d ", ret[i][j]);
		printf("\n");
	}
	printf("segfault after?\n");
	

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
void cudaMatMul(int** C, int** A, int** B, int n)
{
	int i, j, k = 0;	

	// mat mul
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			for(k = 0; k < n; k++)
				C[i][j] += A[i][k] * B[k][j];
}

