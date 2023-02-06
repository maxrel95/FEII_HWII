import numpy as np
import pandas as pd
from statsmodels.sandbox.regression import gmm
from scipy.stats import norm
from numpy.linalg import inv, cholesky


b = np.array( [ .003, .033 ] ).T
A = np.array( [[ .073, .620],
                [ .015, -.122]])

w11sq = .013
w22sq = .0016
rho = .492
w12 = rho * np.sqrt( w11sq * w22sq )
Om = np.array( [
    [ w11sq, w12 ],
    [ w12, w22sq ]
])

n_sim = 500
beta = .95
tau = 2
tol = .0001
max_iter = 1000

Nv = np.array( [ 10, 10 ] )
Ns = np.prod( Nv.T )
P = 3

r = Om.shape[ 0 ]
Omi = inv( Om ) #@ np.eye( r )
B = cholesky( Omi,  ).T
B @ Om @ B.T
Bi = inv( B ) #@ np.eye( r )
F = B @ A @ Bi

X = np.eye( r**2 ) - np.kron( F, F )
Xi = inv( X ) #@ np.eye( r**2 )
SigSt = ( Xi @ np.eye( r ).reshape( [ r**2, 1 ] ) ).reshape( [ r, r ]) 
z1L = -P * np.sqrt( SigSt[ 0, 0 ] )
z1U = P * np.sqrt( SigSt[ 0, 0 ] )
z2L = -P * np.sqrt( SigSt[ 1, 1 ] )
z2U = P * np.sqrt( SigSt[ 1, 1 ] )
NvL = Nv + 1
dz1 = ( z1U - z1L ) / ( NvL[ 0 ] - 1 )
dz2 = ( z2U - z2L ) / ( NvL[ 1 ] - 1 )
z1BG = np.arange( z1L, z1U+dz1, dz1 ).T
z2BG = np.arange( z2L, z2U+dz2, dz2 ).T
z1g = ( z1BG[ 1: ] + z1BG[ :-1 ] ) / 2
z2g = ( z2BG[ 1: ] + z2BG[ :-1 ] ) / 2
z1BG[ 0 ] = -12
z1BG[ -1 ] = 12
z2BG[ 0 ] = -12
z2BG[ -1] = 12

e1 = np.ones( ( Nv[ 0 ], 1 ) )
e2 = np.ones( ( Nv[ 1 ], 1 ) )
zg = np.hstack( [ 
    np.kron( z1g.reshape( [ -1, 1] ), e2 ),
    np.kron( e1, z2g.reshape( [ -1, 1] ) )
])




