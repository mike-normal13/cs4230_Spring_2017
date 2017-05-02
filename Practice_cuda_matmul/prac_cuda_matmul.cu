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
		A[i] = (int*) malloc(N * sizeof(int));
		B[i] = (int*) malloc(N * sizeof(int));
		C[i] = (int*) malloc(N * sizeof(int));

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

			ret[i][j] = 0;

			//printf("%d ", B[i][j]);
		}
		//printf("\n");
	}

	// COPY TO device memory
	// for(i = 0; i < N; i++)
	// {
	// 	for(j = 0; j < N; j++)
	// 	{
	// 		cudaMemcpy((void*)A_c[i][j], (void*)A[i][j], sizeof(int), cudaMemcpyHostToDevice);
	// 		cudaMemcpy((void*)B_c[i][j], (void*)B[i][j], sizeof(int), cudaMemcpyHostToDevice);
	// 		cudaMemcpy((void*)C_c[i][j], (void*)C[i][j], sizeof(int), cudaMemcpyHostToDevice);
	// 	}
	// }

	//cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
    //Copies data between host and device. 

	cudaMemcpy2D(A_c, N * sizeof(int), A, N * sizeof(int), N * sizeof(int), N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy2D(B_c, N * sizeof(int), B, N * sizeof(int), N * sizeof(int), N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy2D(C_c, N * sizeof(int), C, N * sizeof(int), N * sizeof(int), N * sizeof(int), cudaMemcpyHostToDevice);

	cudaMatMul<<<1, 1>>>(C_c, A_c, B_c, N);	

	// for(i = 0; i < N; i++)
	// {
	// 	for(j = 0; j < N; j++)
	// 	{
	// 		cudaMemcpy((void*)ret[i][j], (void*)C_c[i][j], sizeof(int), cudaMemcpyDeviceToHost);
	// 	}
	// }

	cudaMemcpy2D(ret, N * sizeof(int), C_c, N * sizeof(int), N * sizeof(int), N * sizeof(int), cudaMemcpyDeviceToHost);

	// printf("segfault before?\n");
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

		cudaFree(A_c[i]);
		cudaFree(B_c[i]);
		cudaFree(C_c[i]);

		free(ret[i]);
	}

	return 0;
}

extern __global__ 
void cudaMatMul(int** C, int** A, int** B, int n)
{
	int i = 0;	
	int j = 0;
	int k = 0;

	// mat mul
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			for(k = 0; k < n; k++)
				C[i][j] += A[i][k] * B[k][j];
}

