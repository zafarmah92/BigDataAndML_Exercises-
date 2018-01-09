from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def fillMatrix(data):
    final_mat = np.matrix(np.zeros((8,3)))
    
    print final_mat
    c = 0
    for i in range(0,4):
        print 'counter c', c
        for j in range(0,2):
            for k in range(0,3):

                print i+1,' ',j,' ',k, '   data :  ', data[i+1][j][k]
                
                final_mat[c,k] = data[i+1][j][k]
            c += 1
    
    print final_mat 
    






if rank == 0:
   data = [(x+1)**x for x in range(size)]
   print 'we will be scattering:',data
else:
   data = None
   
data = comm.scatter(data, root=0)

if rank == 1 :
    data = np.random.randint(1,2,size=(2,3))

elif rank == 2 :
    data = np.random.randint(2,3,size=(2,3))
elif rank == 3 :
    data = np.random.randint(3,4,size=(2,3))
elif rank == 4 :
    data = np.random.randint(4,5,size=(3,3))
    

print 'rank',rank,'has data:\n',data

newData = comm.gather(data,root=0)

if rank == 0:
   print 'master:\n', newData
   
   another_array = [[]]
   print newData[1][0][1]
   print newData[2]
   print newData[3]
   print newData[4]
   another_array.append(newData[1])
   another_array.append(newData[2])
   another_array.append(newData[3])
   another_array.append(newData[4])
 
 
 
   print 'another array\n ', another_array

   fillMatrix(newData)

#finalMat = comm.gather