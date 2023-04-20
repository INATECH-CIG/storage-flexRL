#%%
import pypsa
import pandas as pd
from matplotlib import pyplot as plt

#%% Reading data
generators = pd.read_csv('input/storage_paper/case_04/FPP_DE.csv', index_col=0)
load = pd.read_csv('input/storage_paper/case_04/IED_DE.csv', index_col=0)
storage_units = pd.read_csv('input/storage_paper/case_04/STO_DE.csv', index_col=0)
cross_border_exchange = pd.read_csv('input/storage_paper/case_04/CBT_DE.csv', index_col=0)
renewable_generation = pd.read_csv('input/storage_paper/case_04/FES_DE.csv', index_col=0)
fuel_prices = pd.read_csv('input/storage_paper/case_04/Fuel.csv', index_col=0)
emissions_factor = pd.read_csv('input/storage_paper/case_04/EmissionFactors.csv', index_col=0)

# %%
def calculate_marginal_cost(pp_info):
       # sourcery skip: inline-immediately-returned-variable
       """
       Parameters
       ----------
       t : timestamp
       Defines the fuel price and CO2 prices at that timestep.
       efficiency_dependence : Bool
       DESCRIPTION.
       passed_capacity : float
       Specified the current power level, required to .

       Returns
       -------
       marginal_cost : TYPE
       DESCRIPTION.
       """

       fuel_price = fuel_prices[pp_info['fuel']]
       co2_price = fuel_prices['co2']

       # Efficiency dependent marginal cost

       # Partial load efficiency dependent marginal costs
       marginal_cost = fuel_price/pp_info['efficiency'] + co2_price * \
              pp_info['emission']/pp_info['efficiency'] + pp_info['variableCosts']
       
       return marginal_cost

def assign_emissions(pp):
     return emissions_factor.loc[pp['fuel'], 'emissions']

def downscale(df, freq):
    df = df.copy()
    df = df.set_index(pd.date_range('2019', periods=len(df), freq='15T'))
    df = df.resample(freq).mean()
    return df


#%% Preprocessing data
index = pd.date_range(start='2019-03-01', end='2019-05-01', freq='H')[:-1]

#Calculating start up costs
generators['start_up_cost'] = generators['warmStartCosts'] * generators['maxPower']

# Downscaling to 1 hour
fuel_prices = downscale(fuel_prices, 'H').loc[index]
load = downscale(load, 'H').loc[index]
cross_border_exchange = downscale(cross_border_exchange, 'H').loc[index]
renewable_generation = downscale(renewable_generation, 'H').loc[index]
renewable_generation.columns = renewable_generation.columns.map(lambda x: x[:-5])

# Calculating max hours
storage_units['max_hours'] = (storage_units['max_soc']-storage_units['min_soc']) / storage_units['max_power_dis']

# calculating renewable max_pu
renewable_generation_p_nom = renewable_generation.max()
renewable_generation = renewable_generation / renewable_generation.max()

# assign emissions factor to generators dataframe
generators['emission'] = generators.apply(lambda x: assign_emissions(x), axis=1)

# Calculating Marginal cost
marginal_cost_df = pd.DataFrame(index=index, columns=generators.index, data=0.)
marginal_cost_df = generators.apply(lambda x: calculate_marginal_cost(x), axis=1).T
marginal_cost_df.index = index

#%%
n = pypsa.Network()
n.set_snapshots(index)

# Adding one bus for Germany
n.add("Bus", "DE")

# Adding generators
n.madd('Generator',
       names=generators.index,
       bus='DE',
       p_nom=generators['maxPower'],
       marginal_cost=marginal_cost_df,
       carrier=generators['technology'],
       committable=True,
       min_up_time=generators['minOperatingTime'],
       min_down_time=generators['minDowntime'],
       start_up_cost=generators['start_up_cost'],
       ramp_limit_up= generators['rampUp'],
       ramp_limit_down= generators['rampDown'])

# Adding loads
n.add('Load', 'DE_demand', bus='DE', p_set=load['demand'])

# Adding storage units
n.madd('StorageUnit',
       storage_units.index,
       bus='DE',
       carrier='technology',
       p_nom=storage_units['max_power_dis'],
       efficiency_store= storage_units['efficiency_ch'],
       efficiency_dispatch= storage_units['efficiency_dis'],
       max_hours=storage_units['max_hours'], 
       inflow=storage_units['natural_inflow'],
       marginal_cost=storage_units['variable_cost_ch'],
       )

# Adding import export
cross_border_exchange['Net_export'] = cross_border_exchange['Export'] - cross_border_exchange['Import']
n.add('Load', 'DE_net_export', bus='DE', p_set=cross_border_exchange['Net_export'])

# Adding VRE
for carrier in renewable_generation.columns:
    n.add('Generator',
           carrier,
           bus='DE',
           p_nom=renewable_generation_p_nom[carrier],
           carrier=carrier,
           p_max_pu=renewable_generation[carrier])

#%%
model = pypsa.opf.network_lopf_build_model(n, n.snapshots)

opt = pypsa.opf.network_lopf_prepare_solver(n, solver_name="gurobi")
opt.solve(model).write()

# model.dual.pprint()
# model.generator_status.pprint()

model.generator_status.fix()
model.state_of_charge.fix()
n.results = opt.solve(model)
n.results.write()

pypsa.opf.extract_optimisation_results(n, n.snapshots)

# %%
# get prices
prices = n.buses_t.marginal_price

states_of_charge = n.storage_units_t.state_of_charge

power = n.storage_units_t.p
charge = power[power<0]*(-1)
charge = charge.fillna(0)
discharge = power[power>0]
discharge = discharge.fillna(0)

# %%
profits = pd.DataFrame(index=charge.index, columns=charge.columns, data=1.)

profits = (discharge - charge) * prices.values \
       - storage_units['variable_cost_ch']*charge \
              - storage_units['variable_cost_ch']*discharge

# %%
#save states_of_charge, charge, discharge, and profits inti opt_results folder as csv files
prices.to_csv('opt_results/prices.csv')
states_of_charge.to_csv('opt_results/states_of_charge.csv')
charge.to_csv('opt_results/charge.csv')
discharge.to_csv('opt_results/discharge.csv')

profits.to_csv('opt_results/profits.csv')

# %%
# now import all those files
states_of_charge = pd.read_csv('opt_results/states_of_charge.csv', index_col=0)
charge = pd.read_csv('opt_results/charge.csv', index_col=0)
discharge = pd.read_csv('opt_results/discharge.csv', index_col=0)
profits = pd.read_csv('opt_results/profits.csv', index_col=0)

# %%
total_profit = profits.sum(axis=0)/1000

# plot the total profit of each storage unit
total_profit.plot(kind='bar')

# %%
#plot prices on a line chart
fig, ax1 = plt.subplots()

#plot prices one one y axis and plot profits on another y axis
ax1.plot(prices, color='blue')
ax1.set_ylabel('prices', color='blue')

# create a second y-axis
ax2 = ax1.twinx()

# plot profits on the second y-axis
ax2.plot(power['st_05'].values, color='red')
ax2.set_ylabel('profits', color='red')

plt.show()
