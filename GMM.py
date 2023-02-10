import numpy as np 
import pandas as pd
from statsmodels.sandbox.regression import gmm


df = pd.read_csv( "GMMData.csv", index_col=0, parse_dates=True )
data = pd.DataFrame()

data[ "realConsGrowth" ] = np.log(df[ 'Consumption' ] / df[ 'CPI' ]).diff().dropna()
data[ "RealSP500" ] = ((df[ "SP500" ] + df[ "Dividend" ])/df[ "CPI" ]).pct_change().dropna()
data[ 'realIr' ] = ( 1 + df[ 'LTIR' ] / 100 )**( 1/12 ) / ( 1 + df[ 'CPI' ].pct_change().dropna()) - 1

df[ 'sp500+div' ] = df[ 'SP500' ] + df[ 'Dividend' ]
df[ 'Rt' ] = df[ 'sp500+div' ] / df[ 'sp500+div' ].shift( 1 )
df[ 'r_lag1' ] = df[ 'Rt' ].shift( 1 )
df[ 'r_lag2' ] = df[ 'Rt' ].shift( 2 )
df[ 'ct' ] = df[ 'Consumption' ] / df[ 'Consumption' ].shift( 1 )
df[ 'ct_lag1' ] = df[ 'ct' ].shift( 1 )
df[ 'ct_lag2' ] = df[ 'ct' ].shift( 2 )
df[ 'const' ] = 1
df[ 'Rft'] = ( 1 + df[ 'LTIR' ] / 100 )**( 1 / 12 )
df[ 'rf_lag1' ] = df[ 'Rft' ].shift( 1 )
df[ 'rf_lag2' ] = df[ 'Rft' ].shift( 2 )
df.dropna( axis=0, inplace=True )


exog_df = df[ [ 'Rt', 'Rft', 'ct' ] ]
instrument_df = df[ [ 'r_lag1', 'r_lag2', 'rf_lag1', 'rf_lag2', 'ct_lag1',
                           'const']]
#instrument_df = df[ [ 'const', 'r_lag1', 'rf_lag1', 'ct_lag1' ] ]
#instrument_df = df[ [ 'const', 'r_lag1', 'r_lag2', 'rf_lag1', 'rf_lag2', 'ct_lag1', 'ct_lag2' ] ]

exog, instrument  = map( np.asarray, [ exog_df, instrument_df ] )
instrument_augmented = np.kron( np.ones( ( 2, 1 ) ), instrument )

c = np.kron( np.ones( ( 2, 1 ) ), exog[ :, -1 ].reshape( [ -1, 1 ] ) )
exog = np.hstack( [ np.hstack( ( exog[ :, 0 ], exog[ :, 1 ] ) ).reshape( [ -1, 1 ] ), c ] )

def moment_condition( params, exog ):
    beta, gamma = params
    rt, ct = exog.T  # unwrap iterable (ndarray)
    
    # moment condition without instrument    
    err1 = beta * rt * np.power( ct,  -gamma ) - 1
    return err1

endog1 = np.zeros( exog.shape[ 0 ] )   
mod1 = gmm.NonlinearIVGMM( endog1, exog, instrument_augmented, moment_condition, k_moms=8, k_params=2 )

w0inv = instrument.T @ instrument / len( endog1 )
param, weights = mod1.fititer( [ 1, -1 ] )
res1 = mod1.fit( [ 1, -1 ], maxiter=100 ) 
print( res1.summary( yname='Euler Eq', xname=[ 'discount', 'CRRA' ] ) )

import matplotlib.pyplot as plt
plt.figure()
#(df[ 'Rt' ]-1).expanding().mean().plot()
#((df[ 'Rt' ]-1)**2).expanding().mean().plot()
#((df[ 'Rt' ]-1)**3).expanding().mean().plot()
((df[ 'Rt' ]-1)**4).expanding().mean().plot()
plt.legend(['1','2', '3', '4'])
