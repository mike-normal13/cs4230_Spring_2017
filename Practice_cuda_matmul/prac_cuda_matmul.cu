#include <stdio.h>
#include <stdlib.h>

#define N 16

extern __global__ 
void cudaMatMul(int C[N][N], int A[N][N], int B[N][N], int n);

int main(int argc, char** argv)
{
	int A[N][N];
	int* B[N][N];

	// result
	int* C[N][N];

	// cuda guys
	int* A_c[N][N];
	int* B_c[N][N];
	int* C_c[N][N];

	// cuda result placed in this value
	int* ret[N][N];

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
			//ret[i][j] = 0;
			//printf("%d ", B[i][j]);
		}
		//printf("\n");
	}

	// COPY TO device memory
	for(i = 0; i < N; i++)
	{
		cudaMemcpy(A_c[i], A[i], N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(B_c[i], B[i], N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(C_c[i], C[i], N * sizeof(int), cudaMemcpyHostToDevice);
	}

	//cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
    //Copies data between host and device. 

	//dim3 dimBlock(N, N);

	cudaMatMul<<<1, 1>>>(C_c, A_c, B_c, N);	

	for(i = 0; i < N; i++)
	{
		cudaMemcpy(ret[i], C_c[i], N * sizeof(int), cudaMemcpyDeviceToHost);
	}

	// printf("segfault before?\n");
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			printf("%d ", ret[i][j]);
		printf("\n");
	}
	fflush(stdout);

	// free arrays
	for(i = 0; i < N; i++)
	{
		free(A[i]);
		free(B[i]);
		free(C[i]);
		cudaFree(A_c[i]);
		cudaFree(B_c[i]);
		cudaFree(C_c[i]);
		free(ret[i]);
	}

	return 0;
}

extern __global__ 
void cudaMatMul(int c[N][N], int a[N][N], int b[N][N], int n)
{
	int i = 0;	
	int j = 0;
	int k = 0;

	// mat mul
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			for(k = 0; k < n; k++)
				c[i][j] += a[i][k] * b[k][j];
}

