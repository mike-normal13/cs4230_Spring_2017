#include <stdio.h>

#define N 4

int main()	
{
 	int A[N*N];
 	int B[N*N];

 	for(int i = 0; i < N * N; i++)
 	{
 		B[i] = i;
 		A[i] = 0;
 	}

	for(int i = 0; i < N; i++)	
	{
		for(int j = 0; j < N * N; j += N)	
		{
			A[j + i] = B[(((j + i) + (N * i)) % (N * N))];		
		}
	}

	for(int k = 0; k < N * N; k++)
	{
		printf("%d -> %d\n", k, A[k]);
	}
}