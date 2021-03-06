#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include <helper_cuda.h>

//  https://gist.github.com/jefflarkin/5390993
//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0);   \
 }                                                                 \
}

#define CHECK(call)                   \ 
{                                                       \ 
  const cudaError_t error = call;                               \ 
  if (error != cudaSuccess)                                                 \ 
    {                                                                         \ 
      printf("Error: %s:%d, ", __FILE__, __LINE__);                             \ 
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));        \ 
      exit(1);                                                                  \ 
    }                                                                           \
}


extern int cudaMemcpy();
extern int cudaFree();

extern __global__             void spmv(int nr_c, int* ptr_c, float* t_c, float* data_c, float* b_c, int* indices_c);

__global__  void spmv_csr_scalar_kernel(int nr_c, int* ptr_c, float* t_c, float* data_c, float* b_c, int* indices_c);

int compare(float *a, float *b, int size, double threshold);

main (int argc, char **argv) 
{
  FILE *fp;
  char line[1024]; 
  int *ptr, *indices;
  float *data, *b, *t;
  int i,j;
  int n; // number of nonzero elements in data
  int nr; // number of rows in matrix
  int nc; // number of columns in matrix

// ------------------- Variables for cuda solution -----------------
  //FILE* fp_c;
  int* ptr_c;
  int* indices_c;
  float* data_c;
  float* b_c;
  float* t_c;
  float* res; // we plan on memcopying t_c to this after the call to spmv returns.
// ------------------- end of Variables for cuda solution -----------------

  // Open input file and read to end of comments
  if (argc !=2) abort(); 

  if ((fp = fopen(argv[1], "r")) == NULL) { abort();  }

  fgets(line, 128, fp);
  while (line[0] == '%') 
  {
    fgets(line, 128, fp); 
  }

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for ptr, indices, data, b and t.
  sscanf(line,"%d %d %d\n", &nr, &nc, &n);
  ptr = (int *) malloc ((nr+1)*sizeof(int));
  indices = (int *) malloc(n*sizeof(int));
  data = (float *) malloc(n*sizeof(float));
  b = (float *) malloc(nc*sizeof(float));
  t = (float *) malloc(nr*sizeof(float));

  //------------ cuda mallocs ------------------------
  cudaMalloc((void**)&ptr_c, (nr+1)*sizeof(int));
  cudaMalloc((void**)&indices_c, n*sizeof(int));
  cudaMalloc((void**)&data_c, n*sizeof(float));
  cudaMalloc((void**)&b_c, nc*sizeof(float));
  cudaMalloc((void**)&t_c, nr*sizeof(float));
  cudaMalloc((void**)&res, nr*sizeof(float));
  //------------ end of cuda mallocs -----------------

  // Read data in coordinate format and initialize sparse matrix
  int lastr=0;

  for (i=0; i<n; i++) 
  {
    int r;
    fscanf(fp,"%d %d %f\n", &r, &(indices[i]), &(data[i]));

    indices[i]--;  // start numbering at 0
    
    if (r!=lastr) 
    { 
      ptr[r-1] = i;
      lastr = r; 
    }
  }

  ptr[nr] = n;

  // initialize t to 0 and b with random data  
  for (i=0; i<nr; i++) 
  {
    t[i] = 0.0;
  }

  for (i=0; i<nc; i++) 
  {
    b[i] = (float) rand()/1111111111;
  }

  cudaMemcpy(indices_c, indices, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_c, ptr, (nr+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_c, b, nc*sizeof(float), cudaMemcpyHostToDevice);

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (i=0; i<nr; i++) 
  {                                                      
    for (j = ptr[i]; j<ptr[i+1]; j++) 
    {
      t[i] = t[i] + data[j] * b[indices[j]];
    }
  }

  // TODO: Compute result on GPU and compare output
  //spmv<<<1, nr>>>(nr, ptr_c, t_c, data_c, b_c, indices_c);
  spmv_csr_scalar_kernel<<<1, nr>>>(nr, ptr_c, t_c, data_c, b_c, indices_c);
  //cudaDeviceSynchronize();
  CHECK(cudaMemcpy(res, t_c, nr*sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError();
  fflush(stdout);
    
  printf("segfault before?\n");
  fflush(stdout);
  
  if(compare(t, res, nr, 0.001))
  {
    printf("sequential and parallel results match!\n");
  }
  printf("segfault after?\n");
  fflush(stdout);
 }

//  http://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf
__global__  void
spmv_csr_scalar_kernel(int nr_c, int* ptr_c, float* t_c, float* data_c, float* b_c, int* indices_c)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < nr_c )
  {
    float dot = 0;
    
    int row_start = ptr_c[row];
    int row_end = ptr_c[row +1];
    
    for(int j = row_start; j < row_end; j++)
    {
      dot += data_c[j] * b_c[indices_c[j]];
    }

    t_c[row] += dot;
  }
}

int compare(float *a, float *b, int size, double threshold) {
    int i;
    for (i=0; i<size; i++) {
      if (abs(a[i] - b[i]) > threshold) return 0;
    }
    return 1;
}

__global__ void spmv(int nr_c, int* ptr_c, float* t_c, float* data_c, float* b_c, int* indices_c)
{
  int i = threadIdx.x;

  //for (i=0; i<nr; i++) 
  if(i < nr_c)
  {                                                      
    for (int j = ptr_c[i]; j<ptr_c[i+1]; j++) 
    {
      t_c[i] = t_c[i] + data_c[j] * b_c[indices_c[j]];
    }
  }
}