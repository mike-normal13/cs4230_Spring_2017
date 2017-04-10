#include <stdio.h>

#define N 4

int main()	
{
 	int A[N*N];
 	int B[N*N];

 	for(int i = 0; i < N * N; i++)
 	{
 		A[i] = i;
 		B[i] = 0;
 	}

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			// for each row of A,
			//	row i is shifted by i elements to the left
			B[j + (N * i)] = A[(N * i) + (((i * N) + (j + i)) % N)];

			//printf("%d -> %d\n", j + i*N, B[j]);			
		}
	}

	for(int i = 0; i < N * N; i++)
	{
		printf("B[%d] = %d\n", i, B[i]);
	}
}