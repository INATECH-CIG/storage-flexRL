# %% 
# import all packages
import json
from datetime import datetime

import pandas as pd
from tqdm.notebook import tqdm

from flexABLE import World

# %% 
# run training on defined scenario
study = 'paper_2'
case = 'st_05'
run = 'run_01'
with open(f'./scenarios/{study}/{case}.json') as f:
    config = json.load(f)
    scenario = config[run]

# %%
# create world
save_policy_dir = 'E:/Codes/flexRL/policies/'

dt = scenario['dt']

snapLength = int(24/dt*scenario['days'])
timeStamps = pd.date_range(f"{scenario['year']}-01-01T00:00:00", f"{scenario['year'] + 1}-01-01T00:00:00", freq='15T')

starting_point = str(scenario['year'])+scenario['start']
starting_point = timeStamps.get_loc(starting_point)

world = World(snapshots=snapLength,
              scenario=scenario['scenario'],
              simulation_id=scenario['id'],
              starting_date=timeStamps[starting_point],
              dt=dt,
              enable_CRM=False,
              enable_DHM=False,
              check_availability=False,
              max_price=scenario['max_price'],
              write_to_db=scenario['write_to_db'],
              rl_mode=scenario['rl_mode'],
              cuda_device=0,
              save_policies=True,
              save_params={'save_dir': save_policy_dir},
              load_params=scenario['load_params'],
              learning_params = scenario['learning_params'])

# %% 
# Load scenario
world.load_scenario(startingPoint=starting_point,
                    importStorages=scenario['import_storages'],
                    opt_storages=scenario['opt_storages'],
                    importCBT=scenario['importCBT'],
                    scale=scenario['scale'])

# %% 
# Start simulation
index = pd.date_range(world.starting_date, periods=len(world.snapshots), freq=f'{str(60 * world.dt)}T')

if world.rl_mode and world.training:
    training_start = datetime.now()
    world.logger.info("################")
    world.logger.info(f'Training started at: {training_start}')

    learning_params = scenario['learning_params']
    training_episodes = learning_params['training_episodes']

    for i_episode in tqdm(range(training_episodes), desc='Training'):
        start = datetime.now()
        world.run_simulation()
        world.episodes_done += 1

        if ((i_episode + 1) % 5 == 0) and world.episodes_done > learning_params['learning_starts']:
            world.training = False
            world.run_simulation()
            world.compare_and_save_policies()
            world.eval_episodes_done += 1
            world.training = True

            if world.write_to_db:
                tempDF = pd.DataFrame(world.mcp, index=index, columns=['Simulation']).astype('float32')
                world.results_writer.writeDataFrame(tempDF, 'Prices', tags={'simulationID': world.simulation_id, "user": "EOM"})

    world.rl_algorithm.save_params(dir_name = 'last_policy')
    for unit in world.rl_powerplants+world.rl_storages:
        unit.save_params(dir_name = 'last_policy')

    training_end = datetime.now()
    world.logger.info("################")
    world.logger.info(f'Training time: {training_end - training_start}')

    if world.write_to_db:
        world.training=False
        world.run_simulation()
        world.results_writer.save_results_to_DB()
        world.logger.info("################")

else:
    start = datetime.now()
    world.run_simulation()
    end = datetime.now()
    world.logger.info(f'Simulation time: {end - start}')

    if world.write_to_db:
        world.results_writer.save_results_to_DB()

if world.rl_mode:
    world.logger.info(f'Average reward: {world.rl_eval_rewards[-1]}')
    world.logger.info(f'Average profit: {world.rl_eval_profits[-1]}')
    world.logger.info(f'Average regret: {world.rl_eval_regrets[-1]}')
else:
    world.logger.info(f'Average reward: {world.conv_eval_rewards[-1]}')
    world.logger.info(f'Average profit: {world.conv_eval_profits[-1]}')
    world.logger.info(f'Average regret: {world.conv_eval_regrets[-1]}')

# %%
world.training=False
for unit in world.rl_powerplants+world.rl_storages:
   unit.load_params(load_params={'id': scenario['id'], 'policy_dir': save_policy_dir, 'dir': 'highest_profit'})
world.run_simulation()
world.logger.info(f'Average reward: {world.rl_eval_rewards[-1]}')
world.logger.info(f'Average profit: {world.rl_eval_profits[-1]}')
world.logger.info(f'Average regret: {world.rl_eval_regrets[-1]}')
world.results_writer.save_results_to_DB()
world.logger.info("################")

# %%
historic_prices = pd.read_csv('input/'+scenario['scenario']+'/prices_DE.csv', index_col = 0)

historic_prices.index = pd.to_datetime(historic_prices.index, unit='ms')
historic_prices = historic_prices[1:]

# %%
historic_prices = pd.read_csv('input/'+scenario['scenario']+'/prices_DE.csv', index_col = 0)
historic_prices.index = pd.to_datetime(historic_prices.index)

# %%
tempDF = pd.DataFrame(
    historic_prices['Historical Price [EUR/MWh]'].values,
    index=pd.date_range(
        f"{scenario['year']}-01-01T00:00:00",
        f"{scenario['year']}-12-31T23:00:00",
        freq='1H',
    ),
    columns=['Historic Price'],
).astype('float32')

world.results_writer.writeDataFrame(tempDF, 'Prices', tags={'simulationID': scenario['id'], 'user': 'EOM'})

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

modelled_prices = pd.DataFrame(index = index, data = world.mcp, columns = ['Modelled MCP'])
#modelled_prices = pd.DataFrame(index = index, data = world.pfc, columns = ['Modelled MCP'])

modelled_prices_hourly = modelled_prices.resample('1H').mean()

historic_prices_hourly = historic_prices.resample('1H').mean()

rmse = mean_squared_error(historic_prices_hourly['Historical Price [EUR/MWh]'],
                          modelled_prices_hourly['Modelled MCP'])**0.5


print('RMSE (1h res):', round(rmse, 2))

mae = mean_absolute_error(historic_prices_hourly['Historical Price [EUR/MWh]'],
                          modelled_prices_hourly['Modelled MCP'])

print('MAE (1h res):', round(mae, 2))

# %%
modelled_prices_daily = modelled_prices.resample('24H').mean()

historic_prices_daily = historic_prices.resample('24H').mean()

rmse = mean_squared_error(historic_prices_daily['Historical Price [EUR/MWh]'],
                          modelled_prices_daily['Modelled MCP'])**0.5


print('RMSE (1d res):', round(rmse, 2))

mae = mean_absolute_error(historic_prices_daily['Historical Price [EUR/MWh]'],
                          modelled_prices_daily['Modelled MCP'])

print('MAE (1d res):', round(mae, 2))
# %%
