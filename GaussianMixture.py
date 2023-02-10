# Maxime Borel 
# Financial Econometrics II 
# Homework II 

import numpy as np 
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


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
        raise NameError( "this function works only for squared matrix" )
    else : 
        ab = np.tril( A ).reshape([ N*K, 1 ], order='f' )
        return ab[ ab != 0 ].reshape( [ -1, 1 ] )

def vech_loop( A ):
    N = A.shape[ 0 ]
    K = A.shape[ 1 ]
    if N != K:
        raise NameError( "this function works only for squared matrix" )
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
sim = np.random.uniform( 0, 1, ( 100, 2 ) )
mu = sim.mean( 0 )
cov = np.cov( sim.T )

def mv_normal_density( x, mu, sigma ):
    # need determinant and inverse of sigma 
    if mu.shape.__len__() == 1:
        mu = mu.reshape( [ -1, 1 ] )

    y = x[ :, 0 ]
    z = x[ :, 1 ]
    sigmaY = np.sqrt( sigma[ 0, 0 ] )
    sigmaZ = np.sqrt( sigma[ 1, 1 ] )
    rho = sigma[ 1, 0 ] / ( sigmaZ * sigmaY )

    first = ( y - mu[ 0 ] ) / sigmaY
    second = ( z - mu[ 1 ] ) / sigmaZ
    third = np.exp( -( 1/(2*( 1 - rho**2 ) ) ) *( first**2 - 2*rho*first*second + second**2 ) )
    cst = 1 / ( 2 * np.pi * sigmaY * sigmaZ * np.sqrt( 1 - rho**2 ))

    density = ( cst * third ).reshape( [ -1, 1 ] )
    return density

test_mvn = mv_normal_density( sim, mu, cov)

# sanity check 
true_mvn = multivariate_normal.pdf(sim, mean=mu, cov=cov)
xd = np.hstack( (test_mvn.reshape(-1,1), true_mvn.reshape(-1,1)))
np.sum( np.abs( test_mvn - true_mvn.reshape( -1, 1 ) ) )

# Question 3
df = pd.read_pickle( 'Data4PhDs.pkl' )
df.head()

plt.figure(1)
plt.scatter( df.iloc[ :, 0 ], df.iloc[ :, 1 ] )
plt.xlabel( df.columns[ 0 ])
plt.ylabel( df.columns[ 1 ])
plt.title( 'Bank dataset' )
plt.savefig( 'results/bankunclassified.png')
plt.show()

T, K = df.shape

X = df.values
mu = X.mean( 0 ).reshape( [ -1, 1 ] )
sigma = np.cov( X.T )

mu1 = mu + 2 * np.random.uniform( size=[ 2, 1 ] )
mu2 = mu - 2 * np.random.uniform( size=[ 2, 1 ] )
sigma1 = sigma.copy()
sigma1[ 0, 0 ] = sigma1[ 0, 0 ] + np.random.uniform( size=[ 1, 1 ] )
sigma2 = sigma.copy()
sigma2[ 1, 1 ] = sigma2[ 1, 1 ] + np.random.uniform( size=[ 1, 1 ] )
smallPi1 = 0.4
smallPi2 = 1 - smallPi1

theta_old = np.vstack( ( 
    mu1.reshape( [ -1, 1 ] ), vech_loop( sigma1 ),
    mu2.reshape( [ -1, 1 ] ), vech_loop( sigma2 ),
    smallPi1 
    )
)

position = 0
e_tol = .0001
max_iter = 1000
dist = 1

while ( dist > e_tol ) & ( position < max_iter ):
    f1 = mv_normal_density( X, mu1, sigma1 )
    f2 = mv_normal_density( X, mu2, sigma2 )
    density = smallPi1*f1 + smallPi2*f2

    log_like = np.log( density ).sum()

    # update proba
    proba_state1 = ( f1*smallPi1 ) / density
    proba_state2 = ( f2*smallPi2 ) / density

    # update mu
    mu1 = ( X*proba_state1 ).sum( 0 )  / proba_state1.sum()
    mu2 = ( X*proba_state2 ).sum( 0 )  / proba_state2.sum()

    # update sigma 
    partial1 = ( X - mu1 ) * np.sqrt( proba_state1 )
    partial2 = ( X - mu2 ) * np.sqrt( proba_state2 )

    sigma1 = ( partial1.T @ partial1 ) / proba_state1.sum( 0 )
    sigma2 = ( partial2.T @ partial2 ) / proba_state2.sum( 0 )

    smallPi1 = proba_state1.mean()
    smallPi2 = 1 - smallPi1

    theta_new = np.vstack( ( 
        mu1.reshape( [ -1, 1 ] ), vech_loop( sigma1 ),
        mu2.reshape( [ -1, 1 ] ), vech_loop( sigma2 ),
        smallPi1
    ))

    dist = ( ( theta_old - theta_new )**2 ).sum()
    theta_old = theta_new
    position += 1
    print( position )

# print results 
print( 'mixeur of gaussian Pi : {}'.format( smallPi1 ) )
print( 'Mean parameters 1st Gaussian : {}'.format( mu1 ) )
print( 'Mean parameters 2nd Gaussian : {}'.format( mu2 ) )
print( 'Covariance parameters 1st Gaussian : {}'.format( sigma1 ) )
print( 'Covariance parameters 2nd Gaussian : {}'.format( sigma2 ) )

gm = GaussianMixture( n_components=2, init_params='random' ).fit( X )

bankType = f1 >= f2

position = 0
for bank in X:
    if bankType[ position ]:
        plt.plot( bank[ 0 ], bank[ 1 ], 'xr' )
    else:
        plt.plot( bank[ 0 ], bank[ 1 ], 'xg' )
    position += 1
plt.xlabel( df.columns[ 0 ] )
plt.ylabel( df.columns[ 1 ] )
plt.title( 'Banks classified by Gaussian Mixture Model' )
plt.savefig( 'results/bankclassified.png' )
plt.show()

