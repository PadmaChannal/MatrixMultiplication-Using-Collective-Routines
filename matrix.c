#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

// Declaration of variables
int rank, size, matSize, i, j; // Rank of processor, size, matrix size and indexes for looping
int** MatrixB, ** MatrixA ; // pointers to dynamically allocate 2D array
int *TempMatrix ; // Temporary array for calculations

int main(int argc , char**argv){	
	
	MPI_Init(&argc , &argv) ;
	MPI_Comm_rank(MPI_COMM_WORLD , &rank) ;
	MPI_Comm_size(MPI_COMM_WORLD , &size) ;
	
	if(rank == 0){	// if you are a process 0 
		
		        FILE *file = fopen("data.txt" , "r" ) ; 
		        fscanf(file , "%d" , &matSize) ;	// read matrix from data.txt

		        MatrixA = malloc(matSize * sizeof(int*)); // Matrix A
		        for(i = 0 ; i < matSize ; i++){
		            MatrixA[i] = malloc(matSize * sizeof(int)) ;
		            for(j = 0; j < matSize ; j++)
		                fscanf(file , "%d" , &MatrixA[i][j]) ;
		        }

		        MatrixB = malloc(matSize * sizeof(int*)); // Matrix B
		        for(i = 0 ; i < matSize ; i++){
		            MatrixB[i] = malloc(matSize * sizeof(int)) ;
		            for(j = 0; j < matSize ; j++)
		                fscanf(file , "%d" , &MatrixB[i][j]) ;
		        }
		        fclose(file);

		// this is a temporary matrix used for calculations. You will understand as you read.        
		TempMatrix = malloc(matSize * matSize * sizeof(int)) ;
		int idx=0 ;
		for(i=0 ; i < matSize ; i++)
			for(j=0 ; j < matSize ; j++)
				TempMatrix[idx++] = MatrixA[i][j] ;

	}

	
	// Broadcast matrix size to all other processes
	MPI_Bcast(&matSize,1,MPI_INT,0,MPI_COMM_WORLD);
	

	// If it is other than process 0, it needs to have its own memory
	if(rank != 0){
		MatrixB = malloc(matSize * sizeof(int*));
		for(i = 0 ; i < matSize ; i++)
			MatrixB[i] = malloc(matSize * sizeof(int)) ;
	}

	// Now broadcast Matrix B to all other matrices.
	for(i = 0 ; i < matSize ; i++)
		MPI_Bcast(MatrixB[i] , matSize , MPI_INT , 0 , MPI_COMM_WORLD) ;
	

	// for the condition n % p == 0. Take extra care of n!=p
	int sentRows , tempRows;
	if(matSize > size){
		sentRows = matSize * (matSize / size) ;
		tempRows = sentRows / matSize ;
	}
	
	else{
		sentRows = matSize ;
		tempRows = sentRows / matSize ;
	}

	// Now scatter the temporary matric 
	int *reciever = malloc(sentRows * sizeof(int)) ;
	MPI_Scatter(TempMatrix , sentRows , MPI_INT , reciever , sentRows , MPI_INT , 0 , MPI_COMM_WORLD);

	int cnt = tempRows , index = 0 , sum=0 , idx=0;
	int* result = malloc(tempRows * matSize * sizeof(int)) ;
	int k , tempIndex;

	// Introduce a barrier, because we want to start the timer
	// when processes actually start matric multiplication

	MPI_Barrier(MPI_COMM_WORLD); 
	double start = MPI_Wtime(); 

	// Matrix multiplication
	for(k = 0 ; k < cnt ; k++){ // cnt is no. of rows in each processes
		tempIndex = k * matSize ;
		for(i=0 ; i < matSize ; i++){
			index = tempIndex ;
			for(j=0 ; j < matSize ; j++)
				sum += reciever[index++] * MatrixB[j][i] ;
			result[idx++] = sum ;
			sum = 0 ;
		}
	}	

	//allocate the memory to collect the result
	int * finalResult = malloc(matSize * tempRows * sizeof(int));
	MPI_Gather(result , matSize * tempRows , MPI_INT , finalResult , matSize * tempRows , MPI_INT , 0 ,MPI_COMM_WORLD);

	// stop the barrier and calculate the time
	MPI_Barrier(MPI_COMM_WORLD); 
	double end = MPI_Wtime(); 
	double time = end - start ;
	printf (" Total time for %d is % d \n" , rank , time ) ;
	

	
	// print the result
	if(rank == 0){
		int tem = 0 ;
		printf("Final result: \n");
		if(size > matSize)
			tem = matSize ;
		else
			tem = size ;
		for(i=0 ; i < tem * (matSize * tempRows) ; i++){
		    printf("%d " , finalResult[i]);
		    if((i+1) % matSize == 0)
		    	printf("\n");
		}

		// Please uncoment this line for execution to know time from 0
		// printf (" FROM p%d is % d \n" , rank , time ) ;

	}

	MPI_Finalize();
	return 0 ;
}