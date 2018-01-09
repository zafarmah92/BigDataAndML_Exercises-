import numpy
from mpi4py import MPI
import mpi4py
import sys
from numpy import source


def sendAll ():
    
    LENGTH = int(sys.argv[1])
    
    rank = comm.Get_rank() # rank of worker
    size = comm.Get_size() # number of workers

    #print 'rank ',rank

    if rank == 0:
        x = numpy.random.randint(0,LENGTH, size = LENGTH)
        #print rank," ",x
        destA = 2*rank + 1
        destB = 2*rank + 2
        
            
        comm.send(x, dest=destA)
        comm.send(x, dest=destB)
            
    else : 
        recvSource = int((rank - 1)/2)
        
        data = comm.recv(source=recvSource)
        #print rank,"  : ", data
        destA = 2*rank + 1
        destB = 2*rank + 2
        
        if (destA < size ):
             
            comm.send(data, dest=destA)
            #comm.Barrier()
        if (destB < size):
            
            comm.send(data, dest = destB)
            #comm.Barrier()
    #MPI.BARRIER()
    comm.Barrier()  
   # comm.barrier(self=rank)
    
    #MPI_BARRIER(MPI_COMM_WORLD)
    #MPI.BARRIER(MPI.COMM_WORLD)


comm = MPI.COMM_WORLD
t_start = MPI.Wtime()
sendAll()
if (comm.Get_rank() == 0) :
   
    t_end = MPI.Wtime() - t_start


    print 'total time ',t_end

#print ('Everthing got distributed',t_end)

