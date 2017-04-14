// compile with mpicc !!!

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

//https://github.com/sgerin/mpi-gemm/blob/master/main.c
//	******many ideas were borrowed from the code above,
//			****I did however try to keep as much code as possible from my sequential solution.

#define N  4
#define P N  // P is the number of processors

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

// scatter may not work
// distributing the data initially:
//		scatterv is dooable,
//			won't be trivial...
//				declare your own data type
//					S.O.!!!				

//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
typedef struct
{
	int nProc;
	MPI_Comm gridComm;
	MPI_Comm rowComm;
	MPI_Comm colComm;
	int rowPosition;
	int colPosition;
	int gridRank;
} GridStruct;

//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
void initGrid(GridStruct* grid);
//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
void cannon(GridStruct* grid, double* block_A, double* block_B, double* block_C);

// circular shift every element of every row in a matrix one position to the left
void shiftALeft(double* a, int dim);
// circular shift every element of every column in a matrix one position up.
void shiftBUp(double* b, int dim);

void skewleft(double* block, int dim);
void skewUp(double* block, int dim);

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank, size;

	GridStruct grid;

	// double C[N * N];
	// double A[N * N];
	// double B[N * N];

	//	matrices
	double* C;
	double* A;
	double* B;

	// blocks
	double* cBlock;
	double* aBlock;
	double* bBlock;

	// place holders for shifted arrays.
	double tempA[N * N];
	double tempB[N * N];

	initGrid(&grid);

	//`https://github.com/sgerin/mpi-gemm/blob/master/main.c
	if(grid.gridRank == 0)
	{
		// allocate memory
		A = (double*) malloc(grid.nProc * grid.nProc * sizeof(double));
		B = (double*) malloc(grid.nProc * grid.nProc * sizeof(double));
		C = (double*) malloc(grid.nProc * grid.nProc * sizeof(double));

		// fill A & B with some data
		
		//for(int i = 0; i < N; i++)
		for(int i = 0; i < grid.nProc; i++)
		{
			for(int j = 0; j < grid.nProc; j++)
			{
				A[i + (N * j)] = i + j;
				B[i + (N * j)] = i * j;
				tempA[i + (N * j)] = 0;// these local guys should be initialized by each process, they are not local at the moment though..
				tempB[i + (N * j)] = 0;// these local guys should be initialized by each process, they are not local at the moment though..
				C[i + (N * j)] = 0;
			}
		}
	}

// 		printf("----A-------\n");
// 	for(int i = 0; i < N * N; i++){	printf("%f\n", A[i]);	}

// printf("----B-------\n");
// 	for(int i = 0; i < N * N; i++){	printf("%f\n", B[i]);	}

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

	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	MPI_Datatype blocktype, type; 

	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	int array_size[2] = {grid.nProc, grid.nProc};
	int subarray_sizes[2] = {(int)sqrt(grid.nProc), (int) sqrt(grid.nProc)};
	int array_start[2] = {0,0};
	
	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start, MPI_ORDER_C, MPI_DOUBLE, &blocktype); 
	MPI_Type_create_resized(blocktype, 0, (int)sqrt(grid.nProc)*sizeof(double), &type);
	MPI_Type_commit(&type);
	
	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	int i, j;
	int displs[grid.nProc];
	int send_counts[grid.nProc];

	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	// For each process, we are allocating a continuous array of doubles. 
	// These are our blocks used in  Cannon's algorithm. 
	aBlock = (double*) malloc(grid.nProc*sizeof(double));
	bBlock = (double*) malloc(grid.nProc*sizeof(double));
	cBlock = (double*) malloc(grid.nProc*sizeof(double));

	// clear block c's cobwebs
	for(i = 0; i < grid.nProc; i++){			cBlock[i] = 0.0;			}

	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	// Setting the offset that we'll use in MPI_Scatterv. 
	// It indicates how much you have to shift in order to distribute the correct blocks to all our processes
	if (grid.gridRank == 0) 
	{
		for(i=0; i<grid.nProc; i++) 
		{
			send_counts[i] = 1;
		}

		int disp = 0;
		for (i=0; i<(int)sqrt(grid.nProc); i++) 
		{
			for (j=0; j<(int)sqrt(grid.nProc); j++) 
			{
				displs[i*(int)sqrt(grid.nProc)+j] = disp;
				disp += 1;
			}
			disp += ((grid.nProc/(int)sqrt(grid.nProc)-1))*(int)sqrt(grid.nProc);
		}
	}

	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	// MPI_Scatterv takes the global matrix A and B then "subdivides" them into blocks that are sent to each
	// process of our communicator MPI_COMM_WORLD
	MPI_Scatterv(A, send_counts, displs, type, aBlock, grid.nProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Scatterv(B, send_counts, displs, type, bBlock, grid.nProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	//MPI_Comm_size( MPI_COMM_WORLD, &size );

	//double myA[N/size][N];

	skewleft(aBlock, (int)sqrt(grid.nProc));
	skewUp(bBlock, (int)sqrt(grid.nProc));

	//void cannon(GridStruct* grid, double* block_A, double* block_B, double* block_C);

	cannon(&grid, aBlock, bBlock, cBlock);
	// now finish the algorithm
	//for(int k = 0; k < N; k++)
	//{
		
	//}

	//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
	// MPI_GATHERV is the inverse operation to MPI_SCATTERV
	// We are building the matrix back from the blocks
	//MPI_Gatherv(block_C, grid.nb_proc,  MPI_FLOAT, mat_C, send_counts, displs, type, 0, MPI_COMM_WORLD);
	MPI_Gatherv(cBlock, grid.nProc,  MPI_DOUBLE, C, send_counts, displs, type, 0, MPI_COMM_WORLD);

	MPI_Finalize();

	for(int i = 0; i < N * N; i++)
	{
		printf("%f\n", C[i]);
	}
}

//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
void initGrid(GridStruct* grid)
{
	//int rank;
    int dims[2];
    int period[2];
    int coords[2];
    int free_coords[2];

    MPI_Comm_size(MPI_COMM_WORLD, &(grid->nProc));
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Seems unused....

    //grid_info->order = grid_info->nb_proc;  // why the need for two variables....?
    //dims[0] = dims[1] = (int) sqrt(grid_info->order);
    //	if we do it this way the smallest matrix we can use is 4X4,
    //		and larger matrices will have to have perfect squares as dimensions
    dims[0] = dims[1] = (int) sqrt(grid->nProc);
    period[0] = period[1] = 1;

    // Create a grid of processus 
    // Store global rank in grid
    // Find the communicators and the coordinates for each processus

    //MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &(grid_info->grid_comm));
    //	TODO: we are setting with 2 dimensions here,
    //			however all of our matrices are 1 dimnesional.....watch out!
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &(grid->gridComm));
    MPI_Comm_rank(grid->gridComm, &(grid->gridRank));
    MPI_Cart_coords(grid->gridComm, grid->gridRank, 2, coords);
    grid->rowPosition = coords[0];
    grid->colPosition = coords[1];

    free_coords[0] = 0; 
    free_coords[1] = 1;
    MPI_Cart_sub(grid->gridComm, free_coords, &(grid->rowComm));

    free_coords[0] = 1; 
    free_coords[1] = 0;
	MPI_Cart_sub(grid->gridComm, free_coords, &(grid->colComm));
}

// shift every row in A one element to the left
void shiftALeft(double* a, int dim)
{
	// for(int i = 0; i < N * N; i++)
	// {
	// 	printf("%f\n", a[i]);
	// }
	//printf("----------- shiftALeft\n");

	//for(int i = 0; i < N; i++)
		for(int i = 0; i < dim; i++)
		{
			double temp = a[i * dim];
			//for(int j = 0; j < (N - 1); j++)
			for(int j = 0; j < (dim - 1); j++)
			{
				//a[j + (i * N)] = a[(j + 1) + (i * N)];
				a[j + (i * dim)] = a[(j + 1) + (i * dim)];
			}
			//a[(N - 1) + (i * N)] = temp;
			a[(dim - 1) + (i * dim)] = temp;
		}
}

// shift every column in B up by one element
void shiftBUp(double* b, int dim)
{
	// for(int i = 0; i < N * N; i++)
	// {
	// 	printf("%f\n", b[i]);
	// }
	// printf("----------- shiftbUp\n");

	//for(int i = 0; i < N; i++)
		for(int i = 0; i < dim; i++)
	{
		double temp = b[i];
		//for(int j = 0; j < N * N; j += N)
		for(int j = 0; j < dim * dim; j += dim)
		{
			//b[i + j] = b[i + (j + N)];
			b[i + j] = b[i + (j + dim)];
		}
		//b[i + (N * N - N)] = temp;
		b[i + (dim * dim - dim)] = temp;
	}
}

//	https://github.com/sgerin/mpi-gemm/blob/master/main.c
//void cannon(GridStruct* grid, double* block_A, double* block_B, double* block_C)
void cannon(GridStruct* grid, double* aBlock, double* bBlock, double* cBlock)
{
	int sqroot = sqrt(grid->nProc);
  	int shift_source, shift_dest;
	MPI_Status status;
	int up_rank, down_rank, left_rank, right_rank;
	int i;

	// Pre-skewing

	// TODO: it looks like instead of shifting the matrices to be mulitplied,
	//			they are shifting the processor grid,
	//				since we already wrote our matrix shifting code we'll see if we can get by without this...
	// MPI_Cart_shift(grid->gridComm, 1, -1, &right_rank, &left_rank); 
	// MPI_Cart_shift(grid->gridComm, 0, -1, &down_rank, &up_rank); 
	// MPI_Cart_shift(grid->gridComm, 1, -grid->rowPosition, &shift_source, &shift_dest); 

	// Execute a blocking send and receive. 
	// The same buffer is used both for the send and for the receive
	// The sent data is replaced by received data
	//MPI_Sendrecv_replace(block_A, sqroot*sqroot, MPI_DOUBLE, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 
	MPI_Sendrecv_replace(aBlock, sqroot*sqroot, MPI_DOUBLE, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 

	// TODO: don't know what this is doing...?
	//MPI_Cart_shift(grid->gridComm, 0, -grid->colPosition, &shift_source, &shift_dest); 
	
	//MPI_Sendrecv_replace(block_B, sqroot*sqroot, MPI_DOUBLE, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 
	MPI_Sendrecv_replace(bBlock, sqroot*sqroot, MPI_DOUBLE, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 
   
    // for (i=0; i<sqroot; i++) 
    // { 
    //     multiply_matrix(grid, block_A, block_B, block_C); 
    //     MPI_Sendrecv_replace(block_A, grid->nProc, MPI_FLOAT, left_rank, 1, right_rank, 1, grid->gridComm, &status); 
    //     MPI_Sendrecv_replace(block_B, grid->nProc, MPI_FLOAT, up_rank, 1, down_rank, 1, grid->gridComm, &status); 
    // }

    // void multiply_matrix(GRID_INFO_T* grid, float* block_A, float* block_B, float* block_C)
// {
// 	// Simple matrix multiplication

// 	int i, j, k;
  //int sqroot = (int)sqrt(grid->nProc);

//     	for (i = 0; i < sqroot; i++)
//         	for (j = 0; j < sqroot; j++)
//             		for (k = 0; k < sqroot; k++)
//                 		block_C[i*sqroot+j] += block_A[i*sqroot+k]*block_B[k*sqroot+j];
// } 

    //for(int i = 0; i < N * N; i++)	// we have to change this
    	for(int i = 0; i < sqroot; i++)	// we have to change this
		{
			//for(int j = 0; j < N; j++)
			for(int j = 0; j < sqroot; j++)
			{
				//for(int k = 0; k < N; k++)
				for(int k = 0; k < sqroot; k++)
				{
					//C[i] = C[i] + tempA[i] * tempB[i];
					cBlock[i*sqroot+j] += aBlock[i*sqroot+k] * bBlock[k*sqroot+j];
					// shift A left
					//shiftALeft(tempA);
					shiftALeft(aBlock, sqroot);
					// shift B up
					//shiftBUp(tempB);
					shiftBUp(bBlock, sqroot);
				}
				
			}
		}
   
   	// Post-skewing

    //MPI_Cart_shift(grid->gridComm, 1, +grid->rowPosition, &shift_source, &shift_dest); 
    //MPI_Sendrecv_replace(block_B, grid->nProc, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 
    MPI_Sendrecv_replace(bBlock, grid->nProc, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 

    //MPI_Cart_shift(grid->gridComm, 0, +grid->colPosition, &shift_source, &shift_dest); 
    //MPI_Sendrecv_replace(block_B, grid->nProc, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->gridComm, &status);
    MPI_Sendrecv_replace(bBlock, grid->nProc, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->gridComm, &status); 	
}

void skewleft(double* mat, int dim)
{
	// skew elements in rows of A left by row number
	//for(int i = 0; i < N; i++)
	for(int i = 0; i < dim; i++)
	{
		//for(int j = 0; j < N; j++)
		for(int j = 0; j < dim; j++)
		{
			//tempA[j + (N * i)] = A[(N * i) + (((i * N) + (j + i)) % N)];
			mat[j + (dim * i)] = mat[(dim * i) + (((i * dim) + (j + i)) % dim)];	
		}
	}
}

void skewUp(double* mat, int dim)
{
	// skew elements in columns of B up by column number
	//for(int i = 0; i < N; i++)	
		for(int i = 0; i < dim; i++)
	{
		//for(int j = 0; j < N * N; j += N)	
			for(int j = 0; j < dim * dim; j += dim)	
		{
			//tempB[j + i] = B[((j + i) + (N * i)) % (N * N)];
			mat[j + i] = mat[((j + i) + (dim * i)) % (dim * dim)];
		}
	}
}