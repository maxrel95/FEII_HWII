# Maxime Borel 
# Financial Econometrics II 
# Homework II 

import numpy as np 
import pandas as pd
from statsmodels.sandbox.regression.gmm import GMM


df = pd.read_csv( "GMMData.csv", index_col=0, parse_dates=True )
data = pd.DataFrame()

df[ 'const' ] = 1

# Compute the inflation rate
df[ 'inflationRate' ] = df[ 'CPI' ].pct_change()

# compute the dividend adjusted sp500 and return 
df[ 'sp500+div' ] = df[ 'SP500' ] + df[ 'Dividend' ]/12
df[ 'Rt' ] = df[ 'sp500+div' ] / df[ 'SP500' ].shift( 1 )

# Compute the real sp500 without and with div adjustement 
df[ 'sp500_real' ] = df[ 'SP500' ] / df[ 'CPI' ]
df[ 'sp500+div_real' ] = ( df[ 'SP500' ] + df[ 'Dividend' ]/12 ) / df[ 'CPI' ]

# compute real rate of return 
df[ 'real_Rt' ] = df[ 'sp500+div_real' ] / df[ 'sp500_real' ].shift( 1 )

# Compute instrument for risky asset
df[ 'r_lag1' ] = df[ 'Rt' ].shift( 1 )
df[ 'real_r_lag1' ] = df[ 'real_Rt' ].shift( 1 )

# Consumption growth rte
df[ 'ct' ] = df[ 'Consumption' ] / df[ 'Consumption' ].shift( 1 )

# real consumption 
df[ 'real_c'] = df[ 'Consumption' ] / df[ 'CPI' ]

# real consumption growht rate 
df[ 'real_ct' ] = df[ 'real_c' ] / df[ 'real_c' ].shift( 1 )

# instrument for consumption 
df[ 'ct_lag1' ] = df[ 'ct' ].shift( 1 )
df[ 'real_ct_lag1' ] = df[ 'real_ct' ].shift( 1 )

# Desannualized risk free
df[ 'Rft' ] = ( 1 + df[ 'LTIR' ] / 100 )**( 1 / 12 )

# Compute real gross risk free rate 
df[ 'real_Rft' ] = ( df[ 'Rft'] ) - df[ 'inflationRate' ]

# instrument for risk free
df[ 'rf_lag1' ] = df[ 'Rft' ].shift( 1 )
df[ 'real_rf_lag1' ] = df[ 'real_Rft' ].shift( 1 )

df.dropna( axis=0, inplace=True )

zvar = df[ [ 'const', 'ct_lag1', 'r_lag1', 'rf_lag1' ] ]  # instrument
xvar = df[ [ 'ct', 'Rt', 'Rft' ] ] # exog variables

yvar = np.zeros( xvar.shape[ 0 ] ) # endog variable,not used
xvar = np.array( xvar )
zvar = np.array( zvar )

class GMMREM( GMM ):

    def momcond(self, params):
        b0, b1 = params
        x = self.exog
        z = self.instrument
        
        # moment condition of stock return 
        m1 = ( z*( b0*( x[ :, 0 ]**( -b1 ) )*x[ :, 1 ] - 1 ).reshape( -1, 1 ) )
        
        # moment condition for risk-free 
        m2 = ( z*( b0*( x[ :, 0 ]**( -b1 ) )*x[ :, 2 ] - 1 ).reshape( -1, 1 ) )
        return np.column_stack(( m1, m2 ))

# 2 Euler functions with 4 instruments in each equation 
model1 = GMMREM( yvar, xvar, zvar, k_moms=8, k_params=2 )
b0 = [ 1, 1 ]
res1 = model1.fit( b0, maxiter=100, optim_method='bfgs' )
print( res1.summary( xname=[ 'beta', 'gamma' ] ) )

with open( 'results/gmmres1.tex', 'w' ) as file:
    file.write( res1.summary( xname=[ 'beta', 'gamma' ] ).tables[ 1 ].as_latex_tabular() )

#### real term 
real_zvar = df[ [ 'const', 'real_ct_lag1', 'real_r_lag1', 'real_rf_lag1' ] ]  # instrument
real_xvar = df[ [ 'real_ct', 'real_Rt', 'real_Rft' ] ] # exog variables

real_xvar = np.array( real_xvar )
real_zvar = np.array( real_zvar )

model2 = GMMREM( yvar, real_xvar, real_zvar, k_moms=8, k_params=2 )
b0 = [ 1, 1 ]
res2 = model2.fit( b0, maxiter=100, optim_method='bfgs' )
print( res2.summary( xname=[ 'beta', 'gamma' ] ) )

with open( 'results/gmmres2.tex', 'w' ) as file:
    file.write( res2.summary( xname=[ 'beta', 'gamma' ] ).tables[ 1 ].as_latex_tabular() )
