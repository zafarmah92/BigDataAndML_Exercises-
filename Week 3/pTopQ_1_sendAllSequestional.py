import numpy
from mpi4py import MPI
import sys


def sendAll ():
    
    LENGTH = int(sys.argv[1])
    
    rank = comm.Get_rank() # rank of worker
    size = comm.Get_size() # number of workers


    if rank == 0:
        x = numpy.random.randint(0,LENGTH, size = LENGTH)
        #print x
        
        for i in range(1,size):
            comm.send(x, dest=i)
            
    else : 
        data = comm.recv(source=0)
        #print rank,"  : ", data


comm = MPI.COMM_WORLD
t_start = MPI.Wtime()
sendAll()
if (comm.Get_rank() == 0) :
   
    t_end = MPI.Wtime() - t_start


    print 'total time ',t_end
"""

if rank == 0:
    x = numpy.linspace(1,100,LENGTH) #vector to be sepreated
    # y  = numpy.linspace(1,100,LENGTH)
    
    print('this is initial array ', x)
    print ' this is Y vector :', y
    
else:
    x = None
    y = None

x_local = numpy.zeros(LENGTH / size )
y_local = numpy.zeros( LENGTH / size )


comm.Scatter(x, x_local, root=0) # Scatter if x Vector
comm.Scatter(y, y_local, root= 0) # Scallter of y vector


sum =  numpy.dot(x_local,y_local)

print 'this is dot Product of process :', rank , '  sum :', sum 



Final_Sum = numpy.array([0.])


comm.Reduce(sum , Final_Sum , op = MPI.SUM )
if (rank == 0):
    print "Final SUM :", Final_Sum[0]

"""
