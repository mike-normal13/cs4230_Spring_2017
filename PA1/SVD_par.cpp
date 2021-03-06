/*************************************************************************************************/
/* SVD Using Jacobis Rotations									*/
/*												*/
/* Compile: g++ -O3 SVD.cpp -o SVD								*/
/* Arguments:											*/
/*												*/
/*	M = # of columns									*/
/*	N = # of Rows										*/
/*												*/
/*	Matrix must be squared (M=N)								*/
/*												*/
/*	-t = print out Timing and # of Iterations						*/
/*	-p = print out Results (U, S, V)							*/
/*	-d = Generate the Octave files for debug and verify correctness				*/
/*												*/
/* Use:	./SVD M N -t -p -d									*/
/*												*/
/* All arguments aren't important, just M and N. If you want, is possible to do 		*/
/* ./SVD M N -t and only print out the timing. As well you can use ./SVD M N -d for debug.	*/
/************************************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include <omp.h>

#define epsilon 1.e-8

using namespace std;

template <typename T> double sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

int main (int argc, char* argv[]){

  int M,N;

  // number of threads
  int num_threads = 16;

#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

  string T,P,Db;
  M = atoi(argv[1]);
  N = atoi(argv[2]);

  double elapsedTime,elapsedTime2;
  timeval start,end,end2;

  if(argc < 4){
	  cout<<"Please input the size of Matrix and at least one of the options: -t -p -d";
	  return 0;
  }

 
  if(M != N){
	  cout<<"Error: Matrix must be square";
	  return 0;
  }
  
  if(argc > 3){

    T = argv[3];
    if(argc > 4){
      P = argv[4];
      if(argc > 5){
        Db = argv[5];
      }
    }
  }
 // cout<<T<<P<<endl;
  
  double **U,**V, *S,**U_t, **V_t, **A;
  double alpha, beta, gamma, c, zeta, t,s,sub_zeta, converge;

  int acum = 0;
  int temp1, temp2;
  converge = 1.0;

  U = new double*[N];
  V = new double*[N];
  U_t = new double*[N];
  V_t = new double*[N];
  A = new double*[N];
  S = new double[N];

  for(int i =0; i<N; i++)
  {
    U[i] = new double[N];
    V[i] = new double[N];
    U_t[i] = new double[N];
    V_t[i] = new double[N];
    A[i] = new double[N];
  }


  //Read from file matrix, if not available, app quit
  //Already transposed

  ifstream matrixfile("matrix");
  if(!(matrixfile.is_open())){
    cout<<"Error: file not found"<<endl;
    return 0;
  }

  for(int i = 0; i < M; i++){
    for(int j =0; j < N; j++){

      matrixfile >> U_t[i][j];
    }
  }

  matrixfile.close();

 
  for(int i=0; i<M;i++){
    for(int j=0; j<N;j++){

      if(i==j){
        V_t[i][j] = 1.0;
      }
      else{
        V_t[i][j] = 0.0;
      }
    }
  }

    //Store A for debug purpouse
  
   for(int i=0; i<M;i++){
      for(int j=0; j<N;j++){

       A[i][j] = U_t[j][i];
      }
    }

  /* SVD using Jacobi algorithm (Sequencial)*/

//********************************************************************************************************************
    gettimeofday(&start, NULL);

    double conv;
    while(converge > epsilon)
   { 		//convergence
   	converge = 0.0;	

    acum++;				//counter of loops

    for(int i = 1; i<M; i++)
    {
      int j = 0;

      //  we need to distribute this "j", whatever that means....
    	for(/*int*/ j = 0; j<i; j++) // we can parallelize this loop, and it is simple....
    	{
    		alpha = 0.0;
    		beta = 0.0;
    		gamma = 0.0;

        int k = 0;

        // leaving this loop alone seemed to be the best thing,
        //  using a parallel for gave incorrect results,
        //    and use reductions slowed things down by about 100 ms on average
        //  we also tried giving each assignment its own section,
        //    which did not seem to affect run time
    		for(/*int*/k = 0; k < N; k++)
    		{
          double utik = U_t[i][k];
          double utjk = U_t[j][k];

             alpha = alpha + (utik * utik);  // bringing this out gave incorrect results
             beta = beta + (utjk* utjk);
             gamma = gamma + (utik * utjk);          
    		}        
            // need to look at moving this line out of the j loop somehow....
            // there is loop carried dependence on "converge" here,
            //    we could use a max reduction here: reduction(max: converge)
          converge = max(converge, abs(gamma)/sqrt(alpha*beta));	//compute convergence
	  								                                         //basicaly is the angle, between column i and j
            // these four lines
        	zeta = (beta - alpha) / (2.0 * gamma);
        	t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));        //compute tan of angle
            // we could make "c" & "s" arrays here....
            //    then parallelize the j loop and we should see a bump in performace,
            //        if we do this we should not have to do the reductionm on "alpha", "beta", and "gamma"
        	c = 1.0 / (sqrt (1.0 + (t*t)));				//extract cos
        	s = c*t;							                //extrac sin
            // if the j loop ended here, could we parallelize this loop
            //  zeta, t, c, and s would need to be private

        #pragma omp parallel private(k,t)
        {
          #pragma omp sections
          {
          //  "the U_t[i][k] will be one time step behind..."
	  		 //Apply rotations on U and V
          #pragma omp section
          for(/*int*/ k=0; k<N; k +=4)
            {
              t = U_t[i][k];
          		U_t[i][k] = c*t - s*U_t[j][k]; // U_t[i][k] depends on j here, if j is less than i 
              ///                                 it remains teh same per iteration
              U_t[j][k] = s*t + c*U_t[j][k]; // there is a dependence on j here... the dependence distance is [0,0] here

              t = U_t[i][k+1];
              U_t[i][k+1] = c*t - s*U_t[j][k+1]; 
              U_t[j][k+1] = s*t + c*U_t[j][k+1]; 

              t = U_t[i][k+2];
              U_t[i][k+2] = c*t - s*U_t[j][k+2]; 
              U_t[j][k+2] = s*t + c*U_t[j][k+2]; 
              
              t = U_t[i][k+3];
              U_t[i][k+3] = c*t - s*U_t[j][k+3]; 
              U_t[j][k+3] = s*t + c*U_t[j][k+3]; 
            }
          

          #pragma omp section
          for(/*int*/ k=0; k<N; k += 4)
            {
              t = V_t[i][k];
              V_t[i][k] = c*t - s*V_t[j][k];
              V_t[j][k] = s*t + c*V_t[j][k];

              t = V_t[i][k+1];
              V_t[i][k+1] = c*t - s*V_t[j][k+1];
              V_t[j][k+1] = s*t + c*V_t[j][k+1];

              t = V_t[i][k+2];
              V_t[i][k+2] = c*t - s*V_t[j][k+2];
              V_t[j][k+2] = s*t + c*V_t[j][k+2];

              t = V_t[i][k+3];
              V_t[i][k+3] = c*t - s*V_t[j][k+3];
              V_t[j][k+3] = s*t + c*V_t[j][k+3];
            }
          }
        }
      }
    }

    
  //Create matrix S
    for(int i = 0; i < M; i++)
    {
   		t=0;
      int j = 0;

      #pragma omp parallel for private(j) firstprivate(i)
      for(j = 0; j < N; j++)
      {
      	t = t + pow(U_t[i][j],2);
      }

      t = sqrt(t);

      //#pragma omp parallel for firstprivate(i, U_t, S) private(j)
      #pragma omp parallel sections firstprivate(i, U_t, S), private(j)
      {
        #pragma omp /*reduction(/ : U_t)*/ section 
        for(j = 0; j < N; j++)
        {
          U_t[i][j] = U_t[i][j] / t;
        }

        #pragma omp section
        for(j = 0; j < N; j++)
        {
      	 //U_t[i][j] = U_t[i][j] / t;
      	 if(i == j)
      	 {
      	   	S[i] = t;
      	 }
        }
      }
    }

    gettimeofday(&end, NULL);
  //********************************************************************************************************************

 /* Develop SVD Using OpenMP */



// fix final result

  for(int i =0; i<M; i++){
    
    for(int j =0; j<N; j++){

      U[i][j] = U_t[j][i];
      V[i][j] = V_t[j][i];
      
    }
    
  }


  //Output time and iterations


  if(T=="-t" || P =="-t"){
    cout<<"iterations: "<<acum<<endl;
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    cout<<"Time: "<<elapsedTime<<" ms."<<endl<<endl;
  }


  // Output the matrixes for debug
  if(T== "-p" || P == "-p"){
  cout<<"U"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){

      cout<<U[i][j]<<"  ";
    }
    cout<<endl;
  }

  cout<<endl<<"V"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){

      cout<<V[i][j]<<"  ";
    }
    cout<<endl;
  }

  cout<<endl<<"S"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){

       if(i==j){  cout<<S[i]<<"  ";}
	
       else{
	       cout<<"0.0  ";
       }
    }
    cout<<endl;
  }

  }

  //Generate Octave files for debug purpouse
   if(Db == "-d" || T == "-d" || P == "-d"){


    ofstream Af;
    //file for Matrix A
    Af.open("matrixA.mat"); 
    Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";


    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Af<<" "<<A[i][j];
      }
      Af<<"\n";
    }
    
    Af.close();

    ofstream Uf;

    //File for Matrix U
    Uf.open("matrixUcpu.mat");
    Uf<<"# Created from debug\n# name: Ucpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";
    
    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Uf<<" "<<U[i][j];
      }
      Uf<<"\n";
    }
    Uf.close();

    ofstream Vf;
    //File for Matrix V
    Vf.open("matrixVcpu.mat");
    Vf<<"# Created from debug\n# name: Vcpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Vf<<" "<<V[i][j];
      }
      Vf<<"\n";
    }
    

    Vf.close();

    ofstream Sf;
    //File for Matrix S
    Sf.open("matrixScpu.mat");
    Sf<<"# Created from debug\n# name: Scpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

    
    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        if(i == j){
         Sf<<" "<<S[i];

        }

        else{
          Sf<<" 0.0";
        }
      }
      Sf<<"\n";
    }
    

    Sf.close();


 }

   delete [] S;
   for(int i = 0; i<N;i++){
	   delete [] A[i];
	   delete [] U[i];
	   delete [] V[i];
	   delete [] U_t[i];
	   delete [] V_t[i];
   }

  return 0;
}
}





