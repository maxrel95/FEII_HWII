import numpy as np 
import pandas as pd
from statsmodels.sandbox.regression.gmm import GMM


df = pd.read_csv( "GMMData.csv", index_col=0, parse_dates=True )
data = pd.DataFrame()

data[ "realConsGrowth" ] = np.log(df[ 'Consumption' ] / df[ 'CPI' ]).diff().dropna()
data[ "RealSP500" ] = ((df[ "SP500" ] + df[ "Dividend" ])/df[ "CPI" ]).pct_change().dropna()
data[ 'realIr' ] = ( 1 + df[ 'LTIR' ] / 100 )**( 1/12 ) / ( 1 + df[ 'CPI' ].pct_change().dropna()) - 1

df[ 'sp500+div' ] = df[ 'SP500' ] + df[ 'Dividend' ]
df[ 'Rt' ] = df[ 'sp500+div' ] / df[ 'sp500+div' ].shift( 1 )
df[ 'r_lag1' ] = df[ 'Rt' ].shift( 1 )
#df[ 'r_lag2' ] = df[ 'Rt' ].shift( 2 )
df[ 'ct' ] = df[ 'Consumption' ] / df[ 'Consumption' ].shift( 1 )
df[ 'ct_lag1' ] = df[ 'ct' ].shift( 1 )
#df[ 'ct_lag2' ] = df[ 'ct' ].shift( 2 )
df[ 'const' ] = 1
df[ 'Rft'] = ( 1 + df[ 'LTIR' ] / 100 )**( 1 / 12 )
df[ 'rf_lag1' ] = df[ 'Rft' ].shift( 1 )
#df[ 'rf_lag2' ] = df[ 'Rft' ].shift( 2 )
df.dropna( axis=0, inplace=True )

#zvar = df[ [ 'const', 'r_lag1', 'r_lag2', 'ct_lag1', 'ct_lag2', 'rf_lag1', 'rf_lag2']]  # instrument
zvar = df[ [ 'const', 'ct_lag1', 'r_lag1', 'rf_lag1']]  # instrument
xvar = df[ ['ct', 'Rt', 'Rft' ] ] # exog variables
# exog, instrument = map(np.asarray, [x[:-1], z])
yvar = np.zeros( xvar.shape[0] ) # endog variable,not used
xvar = np.array( xvar )
zvar = np.array( zvar )

class GMMREM( GMM ):

    def momcond(self, params):
        b0, b1 = params
        x = self.exog
        z = self.instrument
        m1 = ( z*( b0*( x[ :, 0 ]**( b1 ) )*x[ :, 1 ] - 1 ).reshape( -1, 1 ) )
        m2 = ( z*( b0*( x[ :, 0 ]**( b1 ) )*x[ :, 2 ] - 1 ).reshape( -1, 1 ) )
        return np.column_stack(( m1, m2 ))

# 2 Euler functions with 4 instruments in each equation 
model1 = GMMREM(yvar, xvar, zvar, k_moms=8, k_params=2)
b0 = [ 1, -1 ]
res1 = model1.fit( b0, maxiter=100, optim_method='bfgs' )
# param, weights = model1.fititer( [ 1, -1 ], maxiter=100, optim_method='bfgs' )
print(res1.summary( xname=[ 'beta', 'gamma' ] ) )
