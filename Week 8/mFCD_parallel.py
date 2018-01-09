import numpy as np


def matrix_factorization_cd (R,P,Q,K):
    reference = R
    new_arr = np.dot(P,Q.T) 
    print(new_arr)
    T = 5
    lemda = 0.5
    
    for epochs in range(0,100) :
	    for t in range(0,K):
	    	temp_p = P[:,t] ## Broadcast latent of user
	    	temp_q = Q[:,t]	## Broadcast latent of item 
	    	
	    	
	    	for t_i in range(0,len(temp_p)):
	    		
	    	   for t_j in range(0,len(temp_q)):
	    	   
	    		## parallel Update u_star
	    	   
	    	   	u_star = ((reference[t_i][t_j] - temp_p[t_i]*temp_q[t_j] + temp_p[t_i]*temp_q[t_j] ) * temp_q[t_j])/( lemda + np.sum(np.square(temp_q)))
	    		
	    		## parallel update v_star
	    		
	    		
	    	   	v_star = ((reference[t_i][t_j] - temp_p[t_i]*temp_q[t_j] + temp_p[t_i]*temp_q[t_j] ) * temp_p[t_j])/( lemda + np.sum(np.square(temp_p)))
	    	   	

			#update R
		
			reference[t_i][t_j] = reference[t_i][t_j] + temp_p[t_i]*temp_q[t_j] - u_star*v_star 
		
		
	    	   	## update latent fetures 
	    	   	
	    	   	if (u_star != 0): P[t_i,t] = u_star
	    	   	if (v_star != 0 ): Q[t_j,t] = v_star

			#print (u_star , v_star)
    print ("\n\nP ",P)
	    	
 
    return P, Q , new_arr









if __name__ == "__main__":

    R_2 = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

    R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.matrix(np.random.randint(1,5,size=(N,K)))
    Q = np.matrix(np.random.randint(1,5,size=(M,K)))

    #print(len(P), P)
    #print(len(Q), Q)
    
    #nP, nQ = matrix_factorization(R, P, Q, K)
    print("This is P")
    print(P)
    #print("this is Q")
    #print(Q)
    nP, nQ , arr = matrix_factorization_cd(R,P,Q,K)
    nR = np.dot(nP, nQ.T)
    #print(nP) 
    #print(nQ.T)   
    print("what is this ",nR)
    
    print("\n\n")
    print(arr)

