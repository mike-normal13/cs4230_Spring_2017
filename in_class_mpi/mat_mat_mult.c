// compile with mpicc !!!

#include <mpi.h>

#define N = 4

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

	// for(i = 0; i < N: i++)
//		for(j = 0; j < N; j++)
//			c[i][j] = 0
//			for(k = 0; k < N; k++)
//				c[i][j] += A[i][k] * B[k][j];


	MPI_Finalize();

}


// SLURM script
// #SBATCH --account+soc-kp
// #SBATCH --partiton=soc-kp
// #SBATCH --job-name=comp_422_openmp   // <- your job name
// #SBATCH --nodes=2
// #SBATCH --ntasks-per-node=1
// #SBATCH --cpus-per-task=1
// #SBATCH --mem=10g
// #SBATCH --time= 00:10:00
// #SBATCH --export=ALL
// ulimit -c unlimited -s
// mpiexec -n 2 ./matmul


