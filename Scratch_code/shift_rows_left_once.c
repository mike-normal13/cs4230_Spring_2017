#include <stdio.h>

#define N 4

void shiftLeft(double* a);

int main()
{
	double A[N * N];

	for(int i = 0; i < N * N; i++)
 	{
 		A[i] = i;
 	}

	shiftLeft(A);

	for(int i = 0; i < N * N; i++)
	{
		printf("A[%d] =  %f\n", i, A[i]);
	}
}

void shiftLeft(double* a)
{
	for(int i = 0; i < N; i++)
	{
		double temp = a[i * N];
		for(int j = 0; j < (N - 1); j++)
		{
			a[j + (i * N)] = a[(j + 1) + (i * N)];
		}
		a[(N - 1) + (i * N)] = temp;
	}
}