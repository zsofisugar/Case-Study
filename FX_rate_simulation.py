import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy_financial as npf

# I assumed that the volatility is a yearly value, therefore we get the daly by multiplying it by sqrt(number of days in a year)
# We simulate the FX rate change for 5 years, with daily time steps to get the daily Close spot rates

#-----------------------------------------------------------------
#------------------------- Part (a) ------------------------------
#-----------------------------------------------------------------

print('\n     Part (a)\n \nSimulating the GBP/USD spot rates with the GBM model,\
 for 5 years with daily moves (See figure 1)\n')


np.random.seed(1997)        # set seed to for repeatability
d = 252                     # number of trading days in a year
T = d*5                     # final time step (number of days between start and end date)
sigma = 0.093/np.sqrt(d)    # volatility yearly 9.3%

no_paths = 1000             # number of paths to simulate fx spot rate change
S_0 = 1.37625               # initial spot rate
b = np.random.normal(0,sigma,size=(T,no_paths))     # creating the random shocks using np.random.normal()
W = np.cumsum(b, axis=0)                            # summing the random shocks
W = np.insert(W,0,np.zeros(no_paths),axis=0)        # adding day 0 to include it in the plot
gbpusd=S_0*np.exp(W)                                # all simulated paths

# plotting the paths
[plt.plot(gbpusd[:,i]) for i in  np.arange(gbpusd.shape[1])]
plt.xlabel('day'); plt.ylabel('GBP/USD'); plt.title('1000 simulated exchange rate paths')
plt.show()

#-----------------------------------------------------------------
#------------------------- Part (b) ------------------------------
#-----------------------------------------------------------------

print('\n     Part (b)\n \nCalculating IRR using the simulated FX spot rate paths,\
 with the GBP cashflows converted into USD\n')


# fx spot rates at the end of each year (31 Aug) for all paths
fx_spots = pd.DataFrame(gbpusd[[0,d-1,2*d-1,3*d-1,4*d-1,5*d-1]][:])

V_0 = 100000000         # invested amount in GBP
V_proc = 15000000       # yearly returns/proceeds
V_final = V_0 + V_proc  # final return/proceed


Cashflow = pd.DataFrame(-V_0 * fx_spots.iloc[0])  # initial investment of 100,000,000 GBP in USD
for i in range(1,5):
    Cashflow['Proceeds %s'%i] = (V_proc * fx_spots.iloc[i])     # proceeds in USD calculated with spot rate at the time of proceed
Cashflow['Proceeds 5'] = (V_final * fx_spots.iloc[5])
Cashflow['irr'] = Cashflow.apply(npf.irr, axis=1)           # Calculating the IRR for each path

# plotting histogram of IRRs
plt.figure()
bins = np.linspace(-0.01, 0.29, 50)
plt.hist(Cashflow['irr'], bins, histtype='step', label='without option');
plt.xlabel('irr / %');
plt.ylabel('frequency')

# Calculating the required percentiles
perc_without_option=np.percentile(Cashflow['irr'], [95,50,5])
print('IRR of USD without the option:')
print('95th percentile irr', perc_without_option[0], '\n50th percentile irr', perc_without_option[1],
      '\n5th percentile irr',perc_without_option[2])



#-----------------------------------------------------------------
#------------------------- Part (c) ------------------------------
#-----------------------------------------------------------------

print('\n     Part (c)\n \nCalculating the European put option price to hedge the portfolio\n')


K = 1.37625  # strike price of european put option
S_T = (fx_spots.iloc[-1])       # final spot rate of each path at T = 31.08.2026

payoffs = []
for stock_price in S_T:
    payoffs.append(max(K - stock_price, 0))  # finding the payoff of the option for each path

premium = sum(payoffs) / no_paths   # expectation value of payoffs (Black-Scholes), given r=0%
price = 100000000 * premium         # price of option
print('\nEuropean put option to sell 100,000,000GBP at 1.37625 GBP/USD on\
 31-08-2026, purchased on 31-08-2021, has fair option price: {P} USD\
    \n'.format(P=round(price, 2)))


#-----------------------------------------------------------------
#------------------------- Part (d) ------------------------------
#-----------------------------------------------------------------
print('\n     Part (d)\n \nCalculating IRR of the hedged portfolio,\
 including the option premium payment, and the option payoff\n')

Cashflow[0] = Cashflow[0] - price  # option premium paid
Cashflow['Payoffs'] = payoffs
Cashflow['Proceeds 5'] = (V_final * fx_spots.iloc[-1]) + V_0 * Cashflow['Payoffs']  # We add the payoffs to the cashflow
Cashflow.drop(['Payoffs', 'irr'], axis=1, inplace=True)
Cashflow['irr'] = Cashflow.apply(npf.irr, axis=1)       # calculating IRR with put option included

# Plotting histogram of IRRs including put option
plt.hist(Cashflow['irr'], bins, histtype='step', label='with put option');
plt.legend();
plt.title('Distribution of internal rates of return over 1000 simulations')
perc_with_option=np.percentile(Cashflow['irr'], [95,50,5])

print('IRR of USD with European put option:')
print('95th percentile irr', perc_with_option[0], '\n50th percentile irr', perc_with_option[1],
      '\n5th percentile irr',perc_with_option[2])
print('With the use of the put option, the 5th percentile increased from 7% to almost 10%, '
      'which is a great improvement in terms of reducing losses.\nLooking at the distributions, '
      'the one without the option had long tail into the losses, and the Expected shortfall was small and'
      ' may even lead to a negative rate of return,'
      '\nwhereas with the put option the losses are limited, and the worst case IRR is around 7.5%.'
      '\nEven though the 50th percentile of the original distribution is slightly higher, '
      'the spread of the distribution doesnt guarantee a fixed return, \non the other hand the IRR '
      'distribution with the put option is more leptokurtic and right skewed than the other, '
      'making it more likely to return an IRR value of around 13% ')
plt.show()