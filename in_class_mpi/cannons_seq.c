// compile with mpicc !!!

#include <mpi.h>
#include <stdio.h>

//https://github.com/sgerin/mpi-gemm/blob/master/main.c
//	many ideas were borrowed from the code above,
//		I did however try to keep as much code as possible from my sequential solution.

#define N  4
#define P N  // P is the number of processors

void shiftALeft(double* a);
void shiftBUp(double* b);

int main(int argc, char* argv[])
{
	//MPI_Init(&argc, &argv);

	int rank, size;

	double C[N * N];
	double A[N * N];
	double B[N * N];

	// place holders for shifted arrays.
	double tempA[N * N];
	double tempB[N * N];

	//if(rank == 0)
	//{
		// init data
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < N; j++)
			{
				A[i + (N * j)] = i + j;
				B[i + (N * j)] = i * j;
				tempA[i + (N * j)] = 0;// these local guys should be initialized by each process, they are not local at the moment though..
				tempB[i + (N * j)] = 0;// these local guys should be initialized by each process, they are not local at the moment though..
				C[i + (N * j)] = 0;
			}
		}
	//}

// 		printf("----A-------\n");
// 	for(int i = 0; i < N * N; i++)
// 	{
// 		printf("%f\n", A[i]);
// 	}

// printf("----B-------\n");
// 	for(int i = 0; i < N * N; i++)
// 	{
// 		printf("%f\n", B[i]);
// 	}

	// 2X2 should yield:
	//	0	1
	//	0	2

	//	3X3 should yield:
	//	0	5	10
	//	0	8	16
	//	0	11	22

	// multiplication of the above 4X4 default data should yield:
	//	0	14	28	42
	//	0	20	40	60
	//	0	26	52	78
	//	0	32	64	96

	// skew elements in rows of A left by row number
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			tempA[j + (N * i)] = A[(N * i) + (((i * N) + (j + i)) % N)];	
		}
	}

	// skew elements in columns of B up by column number
	for(int i = 0; i < N; i++)	
	{
		for(int j = 0; j < N * N; j += N)	
		{
			tempB[j + i] = B[((j + i) + (N * i)) % (N * N)];
		}
	}

// 	int i, j, k;
//  int sqroot = (int)sqrt(grid->nb_proc);

	//int sqroot = (int)sqrt(N);

//     	for (i = 0; i < sqroot; i++)
//         	for (j = 0; j < sqroot; j++)
//             		for (k = 0; k < sqroot; k++)
//                 		block_C[i*sqroot+j] += block_A[i*sqroot+k]*block_B[k*sqroot+j];
// }

	// now finish the algorithm
	//for(int k = 0; k < N; k++)
	//{
		for(int i = 0; i < N * N; i++)	// we have to change this
		//for(int i = 0; i < sqroot; i++)
		{
			for(int j = 0; j < N; j++)
			//for(int j = 0; j < sqroot; j++)
			{
				//C[i + (N * j)] = C[i + (N * j)] + tempA[i + (N * j)] * tempB[i + (N * j)];
				//C[j + (N * i)] = C[j + (N * i)] + tempA[j + (N * i)] * tempB[j + (N * i)];
					C[i] += tempA[i] * tempB[i];
				// shift A left
					shiftALeft(tempA);
				// shift B up
					shiftBUp(tempB);	
			}
		}
	//}

	//MPI_Finalize();

	for(int i = 0; i < N * N; i++)
	{
		printf("%f\n", C[i]);
	}
}

// shift every row in A one element to the left
void shiftALeft(double* a)
{
	// for(int i = 0; i < N * N; i++)
	// {
	// 	printf("%f\n", a[i]);
	// }

	//printf("----------- shiftALeft\n");

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

// shift every column in B up by one element
void shiftBUp(double* b)
{
	// for(int i = 0; i < N * N; i++)
	// {
	// 	printf("%f\n", b[i]);
	// }

	// printf("----------- shiftbUp\n");

	for(int i = 0; i < N; i++)
	{
		double temp = b[i];
		for(int j = 0; j < N * N; j += N)
		{
			b[i + j] = b[i + (j + N)];
		}
		b[i + (N * N - N)] = temp;
	}
}