import numpy as np
from mpi4py import MPI
import math as m
from decimal import *

import sys


def computerFormula(indx,step) : 

    local_decimal = Decimal(0)
    
    print 'loop start : ',indx,' loop end : ',step
    
    
    for i in xrange(indx,step):
        
       
        
        local_decimal +=  Decimal(1) / Decimal(16**i) *( ( Decimal(4)/Decimal((8*i) + 1) ) - 
                                                        ( Decimal(2)/Decimal((8*i) + 4) ) - 
                                                        ( Decimal(1)/Decimal((8*i) + 5) ) - 
                                                        ( Decimal(1)/Decimal((8*i) + 6) ) ) 

    return local_decimal
    
    
    
    
    
getcontext().prec = 1001

comm = MPI.COMM_WORLD

rank = comm.Get_rank() # rank of worker
size = comm.Get_size() # number of worker's

t_start = MPI.Wtime()  

LENGTH = 0

if (rank == 0):
    
    print 'Master :',rank
    LENGTH = int(sys.argv[1])
  



print 'Slave :',rank

length = comm.bcast(LENGTH, root = 0)

## worker length 

stepSize = int(length/(size-1))

index = 0

for i in range(1,rank):
    index += stepSize
    
result_array = Decimal(0)

if (rank != 0 ) :
    final_index = stepSize+index
    if (rank + 1) == size :
        final_index = length
        
    print 'rank :', rank, ' index : ',index , ' step : ', final_index 
    result_array = np.array(computerFormula(index, final_index))
    
    print 'rank : ', rank , 'result : ',len(str(result_array))


arr = np.zeros(1)
arr[0] = result_array
#arr = np.array(1)
res = Decimal(0)

#arr[0] = result_array

res=comm.reduce(result_array, op=MPI.SUM,root=0)

if (rank == 0) :
    
    print res,' and time :', MPI.Wtime() - t_start
    #result = Decimal(res[0])
    print len(str(res))
    
    
## computing Pi using the function of Compute Formula