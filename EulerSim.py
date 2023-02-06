# Author : Maxime Borel 
# Financial Econometrics II 
# Homework II 

import numpy as np
from scipy.stats import norm, skew, kurtosis
from numpy.linalg import inv, cholesky
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import statsmodels.api as sm

# Question 1 
b = np.array( [ .003, .022 ] ).T
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

imA = inv( np.eye( r ) - A )
yg = zg @ Bi.T + np.kron( b.reshape( [ 1 , -1 ] ) @ imA.T, np.ones( ( Ns, 1 ) ) )
PiM = np.zeros( ( Ns, Ns ) )

for s_idx in range( Ns ):
    ztm1 = zg[ s_idx, : ].reshape( [ -1, 1 ] )
    f = F @ ztm1
    pv1 = norm.cdf( z1BG, loc=f[ 0 ] )
    pv1 = ( pv1[ 1: ] - pv1[ :-1 ] ).reshape( [ -1, 1 ] )

    pv2 = norm.cdf( z2BG, loc=f[ 1 ] )
    pv2 = ( pv2[ 1: ] - pv2[ :-1 ] ).reshape( [ -1, 1 ] )

    PiLine = np.kron( pv1, e2 ) * np.kron( e1, pv2 )
    PiM[ s_idx, : ] = PiLine.T

# get fixed point for value function 
H = np.ones( ( Ns, 1 ) )
is_tol = 0
iter = 0
eyg = np.exp( yg )

while ( ( is_tol < 1 ) & ( iter < max_iter ) ):
    eyg1 = eyg[ :, 0 ].reshape( [ -1, 1 ] )
    eyg2 = eyg[ :, 1 ].reshape( [ -1, 1 ] )
    
    X = ( 1 + H ) * eyg1 * ( eyg2**( -tau ) )
    Hn = beta * PiM @ X
    
    if np.max( np.abs( Hn - H ) ) < tol:
        is_tol = 1
    
    iter += 1
    print( iter )
    H = Hn

rH = H.reshape( [ Nv[ 0 ], Nv[ 0 ] ] ).T
x, y = np.meshgrid(np.arange( 10 ), np.arange( 10 ))

plt.figure( 1 )
ax = plt.axes( projection='3d' )
ax.plot_surface( x , y, rH.T,
 cmap='plasma', alpha=0.8 )
ax.view_init(20, -40) 
plt.title('price/dividend as a function of cons and div gr')
ax.set_xlabel( 'cons g' )
ax.set_ylabel( 'div g' )
ax.set_zlabel( 'PE' )
ax.invert_yaxis()
plt.savefig( 'results/3dFig.png' )
plt.show()

np.random.seed( 1 )
cumPim = np.cumsum( PiM, 1 )
s = np.floor( cumPim.shape[ 0 ] / 2 ) - 1
usim = np.random.uniform( 0, 1, ( n_sim, 1 ) )
res = np.zeros( ( n_sim, 4 ) )

for sim_ctr in range( n_sim ):
    Auxpi = cumPim[ int( s ), : ] < ( usim[ sim_ctr ] * np.ones( ( 1, PiM.shape[ 1 ] ) ) )
    sn = np.sum( Auxpi )
    Rt = ( 1 + H[ sn ] ) / H[ int( s ) ] * eyg[ sn, 0 ]
    res[ sim_ctr, : ] = np.hstack( [ eyg[ sn, : ], Rt, H[ sn ] ] )
    s = sn

plt.figure(2, figsize=(10, 8))
plt.subplot( 4, 1, 1 )
plt.plot( res[ :, 0 ] )
plt.subplot( 4, 1, 2 )
plt.plot( res[ :, 1 ] )
plt.subplot( 4, 1, 3 )
plt.plot( res[ :, 2 ] )
plt.subplot( 4, 1, 4 )
plt.plot( res[ :, 3 ] )
plt.savefig( 'results/subplots.png' )
plt.show()

print( 'Dynamic for dividend growht rate' )
y = np.log( res[ 1:, 0 ] )
T = y.shape[ 0 ] 
x = np.log( res[ :-1, :2 ] )
x = sm.add_constant( x )
mdl = sm.OLS( y, x )
result = mdl.fit()
print( result.summary().as_latex() )

print( 'Dynamic for consumption growht rate' )
y = np.log( res[ 1:, 1 ] )
T = y.shape[ 0 ] 
mdl = sm.OLS( y, x )
result = mdl.fit()
print( result.summary().as_latex() )

y = res[ 1:, 2 ]
T = y.shape[ 0 ] 
x = res[ :-1, -1 ]
x = sm.add_constant( x )
mdl = sm.OLS( y, x )
result = mdl.fit()
print( result.summary().as_latex() )

print( 'Skewness : ', skew( res[ :, 2 ] ) ) 
print( 'Kurtosis : ', kurtosis( res[ :, 2 ] ) + 3 )
