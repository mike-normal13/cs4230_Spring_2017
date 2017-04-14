#include <stdio.h>
#include <stdlib.h>

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
  int* ptr_c;
  int* indices_c;
  float* data_c;
  float* b_c;
  float* t_c;
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
  cudaMalloc(&ptr_c, (nr+1)*sizeof(int));
  cudatMalloc(&indices_c, n*sizeof(int));
  cudatMalloc(&data_c, n*sizeof(float));
  cudatMalloc(&b_c, nc*sizeof(float));
  cudatMalloc(&t_c, nr*sizeof(float));
  //------------ end of cuda mallocs -----------------

  // Read data in coordinate format and initialize sparse matrix
  int lastr=0;

  for (i=0; i<n; i++) 
  {
    int r;
    fscanf(fp,"%d %d %f\n", &r, &(indices[i]), &(data[i]));

    indices_c[i] = indices[i];
    data_c[i] = data[i];

    indices[i]--;  // start numbering at 0

    indices_c[i]--;
    
    if (r!=lastr) 
    { 
      ptr[r-1] = i;
      ptr_c[r - 1] = i;

      lastr = r; 
    }
  }

  ptr[nr] = n;

  ptr_c[nr] = n;

  // initialize t to 0 and b with random data  
  for (i=0; i<nr; i++) 
  {
    t[i] = 0.0;
  }

  for (i=0; i<nc; i++) 
  {
    b[i] = (float) rand()/1111111111;
  }

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (i=0; i<nr; i++) 
  {                                                      
    for (j = ptr[i]; j<ptr[i+1]; j++) 
    {
      t[i] = t[i] + data[j] * b[indices[j]];
    }
  }

  // TODO: Compute result on GPU and compare output
}
