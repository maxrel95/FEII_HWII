# Author : Maxime Borel 
# Financial Econometrics II 
# Homework II 

import numpy as np 
import pandas as pd
from scipy.stats import multivariate_normal

# Problem 2
# Question 1
A = np.array( [
    [ 1, 4, 7 ],
    [ 2, 5, 8 ],
    [ 3, 6, 9 ]
] )

def vech( A ):
    """
    This function implement the vech function without any loop
    """
    N = A.shape[ 0 ]
    K = A.shape[ 1 ]
    if N != K:
        raise NameError( " this function works only for squared matrix")
    else : 
        ab = np.tril( A ).reshape([ N*K, 1 ], order='f' )
        return ab[ ab != 0 ].reshape( [ -1, 1 ] )

def vech_loop( A ):
    N = A.shape[ 0 ]
    K = A.shape[ 1 ]
    if N != K:
        raise NameError( " this function works only for squared matrix")
    else : 
        to_remove = 0
        to_store = np.array( [] )
        for i in range( K ):
            to_store = np.hstack( [ to_store, A[ to_remove:, i ] ] )
            to_remove += 1
        return to_store.reshape( [ -1, 1 ] )


test_vech = vech( A )
print( test_vech )

test_vech = vech_loop( A )
print( test_vech )

# Question 2
sim = np.random.uniform( 0, 1, (100, 2))
mu = sim.mean( 0 )
cov = np.cov( sim.T )

def mv_normal_density( x, mu, sigma ):
    # need determinant and inverse of sigma 
    if mu.shape.__len__() == 1:
        mu = mu.reshape( [ -1, 1 ] )

    N = x.shape[ 1 ]
    det_sigma = sigma[ 0, 0 ]*sigma[ 1, 1 ] - sigma[ 1, 0 ]*sigma[ 0, 1 ]
    inv_sigma = np.array( [
        [ sigma[ 1, 1 ], -sigma[ 0, 1 ]],
        [ -sigma[ 1, 0 ], sigma[ 0, 0 ] ]
    ] )
    inv_sigma = inv_sigma / det_sigma

    y = x[ :, 0 ]
    z = x[ :, 1 ]
    sigmaY = np.sqrt( sigma[ 0, 0 ] )
    sigmaZ = np.sqrt( sigma[ 1, 1 ] )
    rho = sigma[ 1, 0 ] / ( sigmaZ * sigmaY)

    first = ( y - mu[ 0 ] ) / sigmaY
    second = ( z - mu[ 1 ] ) / sigmaZ
    third = np.exp( -0.5*( 1/( 1 - rho**2 ) )*( first**2 - 2*rho*first*second + second**2 ) )
    cst = 1 / ( 2 * np.pi * sigmaY * sigmaZ * np.sqrt( 1 - rho**2 ))

    density = cst * third
    return density.reshape( [ -1, 1 ] )

test_mvn = mv_normal_density( sim, mu, cov)

# sanity check 
true_mvn = multivariate_normal.pdf(sim, mean=mu, cov=cov)
print( np.sum( np.abs( test_mvn - true_mvn ) ) )