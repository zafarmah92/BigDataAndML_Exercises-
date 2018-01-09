import numpy




def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] >= 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T



def matrix_factorization_cd (R,P,Q,K):
    refernce = R 
    T = 5
    lemda = 0.5

    for steps in range(0,T):
        for i in range(0,len(P)):
            #print(P[i])
            temp_p = P[i]
                    
            for j in range(0,len(Q)):

                temp_q = Q[j]

                dot = numpy.dot(temp_p,temp_q.T)
                print("np dot ", dot)

                for k in range(0,K):
                    #print(k)
                    z = ((refernce[i][j] - dot + P[i][k]*Q[j][k] ) * Q[j][k])/(lemda + (Q[j][k]) ** 2)
                    print("This is z",z)
                    R[i][j] = R[i][j] - ((z - P[i][k])*Q[j][k])
                    P[i][k] = z 



                    #print(z)
                    s = ((refernce[i][j] - dot + P[i][k]*Q[j][k] ) * P[j][k])/(lemda + (P[j][k]) ** 2)
                    print("this is s",s)
                    R[i][j] = R[i][j] - ((s - Q[j][k])*P[i][k])
                    Q[j][k] = s



    return P , Q 









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

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.randint(1,5,size=(N,K))
    Q = numpy.random.randint(1,5,size=(M,K))

    #print(len(P), P)
    #print(len(Q), Q)
    
    #nP, nQ = matrix_factorization(R, P, Q, K)
    print(P)
    print("this is Q")
    print(Q)
    nP, nQ = matrix_factorization_cd(R,P,Q,K)
    nR = numpy.dot(nP, nQ.T)
    print(nP) 
    print(nQ.T)   
    print(nR)

