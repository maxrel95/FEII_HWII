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
df[ 'r_lag4' ] = df[ 'Rt' ].shift( 4 )
df[ 'r_lag6' ] = df[ 'Rt' ].shift( 6 )
df[ 'ct' ] = df[ 'Consumption' ] / df[ 'Consumption' ].shift( 1 )
df[ 'ct_lag1' ] = df[ 'ct' ].shift( 1 )
df[ 'ct_lag2' ] = df[ 'ct' ].shift( 2 )
df[ 'ct_lag4' ] = df[ 'ct' ].shift( 4 )
df[ 'ct_lag6' ] = df[ 'ct' ].shift( 6 )
df[ 'const' ] = 1
df[ 'Rft'] = ( 1 + df[ 'LTIR' ] / 100 )**( 1 / 12 )
df[ 'rf_lag1' ] = df[ 'Rft' ].shift( 1 )
df[ 'rf_lag2' ] = df[ 'Rft' ].shift( 2 )
df[ 'rf_lag4' ] = df[ 'Rft' ].shift( 4 )
df[ 'rf_lag6' ] = df[ 'Rft' ].shift( 6 )
df.dropna( axis=0, inplace=True )


exog_df = df[ [ 'Rt', 'Rft', 'ct' ] ]
instrument_df = df[ [ 'r_lag1', 'r_lag2', 'rf_lag1', 'rf_lag2', 'ct_lag1',
                           'const']]

instrument_df1 = df[ [ 'r_lag1', 'rf_lag1', 'ct_lag1', 'const' ] ]                           
instrument_df2 = df[ [ 'r_lag2', 'rf_lag2', 'ct_lag2', 'const' ] ]
instrument_df4 = df[ [ 'r_lag4', 'rf_lag4', 'ct_lag4', 'const' ] ]
instrument_df6 = df[ [ 'r_lag6', 'rf_lag6', 'ct_lag6', 'const' ] ]

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
mod1 = gmm.NonlinearIVGMM( endog1, exog, instrument_augmented, moment_condition )

w0inv = instrument.T @ instrument / len( endog1 )
param, weights = mod1.fititer( [ 1, -1 ] )#, inv_weights=w0inv ) 
res1 = mod1.fit( [ 1, -1 ], maxiter=100 ) #inv_weights=w0invm , ) 
print( res1.summary( yname='Euler Eq', xname=[ 'discount', 'CRRA' ] ) )

#### one lag 
instrument_augmented1 = np.kron( np.ones( ( 2, 1 ) ), instrument_df1.values )
mod2 = gmm.NonlinearIVGMM( endog1, exog, instrument_augmented1, moment_condition )
res2 = mod2.fit( [ 1, -1 ], maxiter=100  ) 

#### one lag 
instrument_augmented2 = np.kron( np.ones( ( 2, 1 ) ), instrument_df2.values )
mod3 = gmm.NonlinearIVGMM( endog1, exog, instrument_augmented2, moment_condition )
res3 = mod3.fit( [ 1, -1 ], maxiter=100  ) 

#### one lag 
instrument_augmented4 = np.kron( np.ones( ( 2, 1 ) ), instrument_df4.values )
mod4 = gmm.NonlinearIVGMM( endog1, exog, instrument_augmented4, moment_condition )
res4 = mod4.fit( [ 1, -1 ], maxiter=100  ) 

#### one lag 
instrument_augmented6 = np.kron( np.ones( ( 2, 1 ) ), instrument_df6.values )
mod5 = gmm.NonlinearIVGMM( endog1, exog, instrument_augmented6, moment_condition )
res5 = mod5.fit( [ 1, -1 ], maxiter=100  ) 

