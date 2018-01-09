import cv2
import numpy as np
from mpi4py import MPI
import sys
from matplotlib import pyplot as plt
from openpyxl.styles.colors import BLUE

comm = MPI.COMM_WORLD

rank = comm.Get_rank() # rank of worker
size = comm.Get_size() # number of

shape = []
img = [] 
t_start = MPI.Wtime()  

if (rank == 0) :
     	
    img = cv2.imread('/media/zfar/media files/beach.jpg',1)
    
img = comm.bcast(img, root = 0)

print 'workers ' , size , 'rank : ', rank,' shape', np.shape(img)

shape = np.shape(img)
step = int(shape[1]/(size-1))

print 'rank ',rank,' step ', step

grey_counter = np.zeros(256)
green_counter = np.zeros(256)
blue_counter = np.zeros(256)
red_counter = np.zeros(256) 

index = 0
for i in range (1,rank):
    index += step


print 'rank ', rank,'index ',index ,' index step ',index+step

local_img = img[:,index:index+step,:]
grey_image = cv2.cvtColor( local_img, cv2.COLOR_RGB2GRAY )

b,g,r = cv2.split(local_img)
    
b_shape = np.shape(b)
    
for i in range(0,b_shape[0]):
    for j in range(0,b_shape[1]):
            
        temp = b[i,j]
        blue_counter[temp] += 1 
        
        temp = g[i,j]
        green_counter[temp] += 1
        
        temp = r[i,j]
        red_counter[temp] += 1  
        
        temp = grey_image[i,j]
        grey_counter[temp] += 1

b_c = np.zeros(256)
g_c = np.zeros(256)
r_c = np.zeros(256)
g_c = np.zeros(256)

comm.Reduce(blue_counter,b_c, op= MPI.SUM, root = 0)
comm.Reduce(green_counter,g_c,op=MPI.SUM , root = 0)
comm.Reduce(red_counter,r_c,op=MPI.SUM, root = 0)
comm.Reduce(grey_counter,g_c,op=MPI.SUM, root = 0)


if (rank == 0):

    t_end = MPI.Wtime() - t_start
    print t_end 
    
    plt.figure(1)
    #ax1 = fig1.add_subplot(111)
    
    plt.plot(b_c,'blue')
    plt.plot(g_c,'green')
    plt.plot(r_c,'red')
   

    plt.figure(2)
    plt.plot(g_c,'grey')
    plt.show()
