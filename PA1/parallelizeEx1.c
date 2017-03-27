#include <stdlib.h>

void main()
{
	int a[10];
	int b[10];
	int c[10];
	int d[10];

	int A[10];
	int B[10];
	int C[10];
	int D[10];

	for(int i = 0; i<10; i++)
	{
		a[i] = 0;
		b[i] = 0;
		c[i] = i;
		d[i] = i;

		A[i] = 0;
		B[i] = 0;
		C[i] = i;
		D[i] = i;
	}

	for(i = 1; i <= 10; i++)
	{
		a[i] = a[i] + b[i];
		b[i + 1] = c[i] + d[i];
	}

	for(i = 1; i <= 10; i++)
	{
		B[i + 1] = C[i] + D[i];
		A[i] = A[i] + B[i];
	}

	for(int i = 0; i < 10; i++)
	{
		assert(a[i] == A[i]);
		assert(b[i] == B[i]);
	}
}