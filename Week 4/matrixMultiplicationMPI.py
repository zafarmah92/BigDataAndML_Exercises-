import numpy as np
from mpi4py import MPI
import sys
from setuptools.command.alias import format_alias

    


def fillMatrix(data,step,matrix_size):
    print 'Fill matrix'
    f_mat = np.matrix(np.zeros((matrix_size,matrix_size)))
    c = 0
    print f_mat
    for i in range(0,matrix_size):
        for j in range(0,step):
            for k in range(0,matrix_size):
                
                print i+1,' ',j,' ',k, '   data :  ', data[i+1][0][0][0]
                
                f_mat[c,k] =  data[i+1][j][k]
            c += 1
    
    print f_mat
   
def matrixMultiple(A,B,r):
    
    print 'Matrix multiple '
    
    C = np.matmul(A,B)
    
    return C

comm = MPI.COMM_WORLD

rank = comm.Get_rank() # rank of worker
size = comm.Get_size() # number of worker's



matrix_size = int(sys.argv[1])

t_start = MPI.Wtime()  
mat_A =[[]]
mat_B = [[]]
mat_C = [[]]

if (rank == 0):
    
    
    print 'Master :',rank,' matrix size :',matrix_size % (size-1)
    
    if (matrix_size % (size-1)) == 0 :
        print "its even we can work with this"
        mat_A = np.matrix(np.random.randint(1,5,size=(matrix_size,matrix_size)))
        mat_B = np.matrix(np.random.randint(1,5,size=(matrix_size,matrix_size))) 
        
        mat_C = np.matrix(np.zeros((matrix_size,matrix_size)))
        
        print mat_A
        print mat_B




mat_B = comm.bcast(mat_B, root = 0)
mat_A = comm.bcast(mat_A, root = 0)

if (matrix_size % (size -1) == 0): 

    print 'Slave :',rank

    shape = np.shape(mat_A) 


    step = int(shape[0]/(size-1))


    index = 0

    for i in range(1,rank):
        index += step
    
    final_step = index+step    

#print 'rank ',rank,' index :',index ,' final index: ',final_step

    temp_mat = mat_A[index:final_step,:]

    result= 0
    final_gather = 0
    if (rank != 0):
        result = matrixMultiple(temp_mat,mat_B,rank)
        
    final_gather = comm.gather( result,root=0)

    if (rank == 0):
        print 'rank : ',rank,' result :',final_gather
        print '\n\n\n ', MPI.Wtime() - t_start 
        #fillMatrix(final_gather, step , matrix_size)
     
else:
    print "Not fully squared matrix & rank :",rank