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

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank;
	int size;

	double C[N][N];
	double A[N][N];
	double B[N][N];

	if(rank == 0)
	{
		// init data
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < N; j++)
			{
				A[i][j] = i + j;
				B[i][j] = i * j;
			}
		}
	}

	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &size );

	double myA[N/size][N];

	MPI_Scatter(A, (N*N)/size, MPI_DOUBLE, myA, (N*N)/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
		{
			C[i][j] = 0;
			for(int k = 0; k < N; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
				printf("%d ", C[i][j]);
			}
			printf("\n");
		}

	MPI_Finalize();
}