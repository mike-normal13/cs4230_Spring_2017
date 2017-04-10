#include <stdio.h>

#define N 4

void shiftUp(double* a);

int main()
{
	double A[N * N];

	for(int i = 0; i < N * N; i++)
 	{
 		A[i] = i;
 	}

	shiftUp(A);

	for(int i = 0; i < N * N; i++)
	{
		printf("A[%d] =  %f\n", i, A[i]);
	}
}

void shiftUp(double* a)
{
	for(int i = 0; i < N; i++)
	{
		double temp = a[i];
		for(int j = 0; j < N * N; j += N)
		{
			a[i + j] = a[i + (j + N)];
		}
		a[i + (N * N - N)] = temp;
	}
}
