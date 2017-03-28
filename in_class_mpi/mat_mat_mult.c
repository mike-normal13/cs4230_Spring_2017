// compile with mpicc !!!

#include <mpi.h>
#include <stdio.h>

#define N  4

// scatter sends adjacent data
// 	with scatter the send bufer is A,
// 	send sendCont: (N * N)/P == SendCount
// 	type MPI_Float or MPI_Double
// 	destination buffer, local copy of part of A that we are currently working on 
//		declare myA = (N*N)/P
//			double myA[N/P][N]
//	Dest count = (N*N)/P
//	Dest type should match above
//	Root = 0
//	MPI_Comm .. WORLD
//	double myB[N][N/P] <- transpose 

// isend and ireceive:
//	proc0 sends its part of A to proc1 and vice versa
// make sure you don't overwrite stuff
//	MPI_scatter() -> then MPI_isend() -> then MPI_ireceive() -> then do some calc1() -> mpi_wait() -> then do calc2()
// with ireceiver, the source is the rank of the proc i'm receiveing from

// for(i = 0; i < N: i++)
//		for(j = 0; j < N; j++)
//			c[i][j] = 0
//			for(k = 0; k < N; k++)
//				c[i][j] += A[i][k] * B[k][j];

int main(int argc, char* argv[])
{
	int rank;
	int size;

	int c[N][N];
	int A[N][N];
	int B[N][N];

	// init data
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			A[i][j] = i + j;
			B[i][j] = i * j;
		}
	}

	MPI_Init(&argc, &argv);

	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &size );

	for(int i = 0; i < N: i++)
		for(int j = 0; j < N; j++)
		{
			c[i][j] = 0
			for(int k = 0; k < N; k++)
			{
				c[i][j] += A[i][k] * B[k][j];
				printf("%d ", c[i][j]);
			}
			printf("\n");
		}


	MPI_Finalize();

}