# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:48:29 2020

@author: INATECH-XX
"""

from datetime import datetime
import pandas as pd
import os

class ResultsWriter():
    def __init__(self,
                 database_name,
                 simulation_id,
                 write_to_db,
                 starting_date='2018-01-01T00:00:00',
                 host='localhost',
                 port=8086,
                 user='root',
                 password='root',
                 world=None):

        self.user = user
        self.password = password
        self.database_name = database_name
        self.simulation_id = simulation_id
        self.starting_date = starting_date
        self.world = world

        self.index = pd.date_range(
            self.world.starting_date,
            periods=len(self.world.snapshots),
            freq=f'{str(60 * self.world.dt)}T'
            )

        if write_to_db:
            from influxdb import InfluxDBClient
            from influxdb import DataFrameClient

            # Creating connection and Database to save results
            self.client = InfluxDBClient(host=host, port=port)
            self.client.create_database(database_name)
            self.client.switch_database(database_name)

            self.dfClient = DataFrameClient(host=host,
                                            port=port,
                                            username=self.user,
                                            password=self.password,
                                            database=self.database_name)
            
            self.dfClient.switch_database(self.database_name)


    def writeDataFrame(self, df, measurement, tags=None):
        if tags is None:
            tags = {'simulationID': 'Historic_Data'}

        self.dfClient.write_points(dataframe=df,
                                   measurement=measurement,
                                   tags=tags,
                                   protocol='line')


    def save_results_to_DB(self):
        start = datetime.now()
        self.world.logger.info('Writing Capacities and Prices to Server - This may take couple of minutes.')

        # writing Merit Order Price
        tempDF = pd.DataFrame(self.world.pfc, index=self.index, columns=['Merit order']).astype('float32')
        self.writeDataFrame(tempDF, 'Prices', tags={'simulationID': self.world.simulation_id, "user": "EOM"})

        # writing EOM market prices
        tempDF = pd.DataFrame(self.world.mcp, index=self.index, columns=['Simulation']).astype('float32')
        self.writeDataFrame(tempDF, 'Prices', tags={'simulationID': self.world.simulation_id, "user": "EOM"})

        # writing EOM demand
        tempDF = pd.DataFrame(self.world.markets['EOM']['EOM_DE'].demand.values(), index=self.index, columns=['EOM demand']).astype('float32')
        self.writeDataFrame(tempDF, 'Demand', tags={'simulationID': self.world.simulation_id, "user": "EOM"})

        # writing residual load
        tempDF = pd.DataFrame(self.world.res_load['demand'].values, index=self.index, columns=['Residual load']).astype('float32')
        self.writeDataFrame(tempDF, 'Demand', tags={'simulationID': self.world.simulation_id, "user": "EOM"})

        # writing residual load forecast
        tempDF = pd.DataFrame(self.world.res_load_forecast['demand'].values, index=self.index, columns=['Residual load forecast']).astype('float32')
        self.writeDataFrame(tempDF, 'Demand', tags={'simulationID': self.world.simulation_id, "user": "EOM"})

        # save total capacities, must-run and flex capacities and corresponding bid prices of power plants
        self.write_pp()
        # write storage capacities
        self.write_storages()

        finished = datetime.now()
        self.world.logger.info(f'Saving into database time: {finished - start}')


    def save_result_to_csv(self):
        self.world.logger.info('Saving results into CSV files...')

        directory = f'output/{self.world.scenario}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory+'/PP_capacities')
            os.makedirs(directory+'/STO_capacities')

        # writing EOM market prices as CSV
        tempDF = pd.DataFrame(self.world.mcp,
                              index=self.index,
                              columns=['Price']).astype('float32')

        tempDF.to_csv(directory + 'EOM_Prices.csv')

        # save total capacities of power plants as CSV
        for powerplant in (self.world.rl_powerplants+self.world.powerplants+self.world.vre_powerplants):
            tempDF = pd.DataFrame(data=powerplant.total_capacity, index=self.index, columns=['Total pp']).astype('float32')
            tempDF.to_csv(directory + f'PP_capacities/{powerplant.name}_Capacity.csv')

        # write storage capacities as CSV
        for storage in (self.world.storages+self.world.rl_storages):
            tempDF = pd.DataFrame(storage.total_capacity, index=self.index, columns=['Total st']).astype('float32')
            tempDF.to_csv(directory + f'STO_capacities/{storage.name}_Capacity.csv')

            tempDF = pd.DataFrame(storage.profits, index=self.index, columns=['Profits']).astype('float32')
            tempDF.to_csv(directory + f'STO_capacities/{storage.name}_Profits.csv')

            tempDF = pd.DataFrame(storage.soc[:-1], index=self.index, columns=['SOC']).astype('float32')
            tempDF.to_csv(directory + f'STO_capacities/{storage.name}_SOC.csv')

        self.world.logger.info('Saving results complete')

    def write_pp(self):
        for powerplant in (self.world.rl_powerplants+self.world.powerplants+self.world.vre_powerplants):
            tags = {'simulationID': self.world.simulation_id,
                    'UnitName': powerplant.name,
                    'Technology': powerplant.technology}

            tempDF = pd.DataFrame(data=powerplant.total_capacity, index=self.index, columns=['Total pp']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Capacities',
                                tags=tags)

            tempDF = pd.DataFrame(powerplant.bids_mr, index=['Capacity_MR', 'Price_MR']).T
            tempDF = tempDF.set_index(self.index).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Capacities',
                                tags=tags)

            tempDF = pd.DataFrame(powerplant.bids_flex, index=['Capacity_Flex', 'Price_Flex']).T
            tempDF = tempDF.set_index(self.index).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Capacities',
                                tags=tags)

            tempDF = pd.DataFrame(powerplant.rewards, index=self.index, columns=['Rewards']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Rewards',
                                tags=tags)

            tempDF = pd.DataFrame(powerplant.regrets, index=self.index, columns=['Regrets']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Regrets',
                                tags=tags)

            tempDF = pd.DataFrame(powerplant.profits, index=self.index, columns=['Profits']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Profits',
                                tags=tags)

    
    def write_storages(self):
        for storage in (self.world.storages+self.world.rl_storages):
            tags = {'simulationID': self.world.simulation_id,
                    'UnitName': storage.name,
                    'Technology': storage.technology}
            
            tags_with_dir = tags.copy()
            
            tempDF = pd.DataFrame(storage.total_capacity, index=self.index, columns=['Total st']).astype('float32')

            tags_with_dir['direction'] = 'discharge'
            self.writeDataFrame(dataframe=tempDF.clip(lower=0),
                                measurement='Capacities',
                                tags=tags_with_dir)

            tags_with_dir['direction'] = 'charge'
            self.writeDataFrame(dataframe=tempDF.clip(upper=0),
                                measurement='Capacities',
                                tags=tags_with_dir)            

            tempDF = pd.DataFrame(storage.bids_supply, index=['Capacity_dis', 'Price_dis']).T
            tempDF = tempDF.set_index(self.index).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Capacities',
                                tags=tags)

            tempDF = pd.DataFrame(storage.bids_demand, index=['Capacity_ch', 'Price_ch']).T
            tempDF = tempDF.set_index(self.index).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Capacities',
                                tags=tags)

            tempDF = pd.DataFrame(storage.rewards, index=self.index, columns=['Rewards']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Rewards',
                                tags=tags)

            tempDF = pd.DataFrame(storage.energy_cost[:-1], index=self.index, columns=['Energy cost']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='Energy cost',
                                tags=tags)

            tempDF = pd.DataFrame(storage.soc[:-1], index=self.index, columns=['SOC']).astype('float32')
            self.writeDataFrame(dataframe=tempDF,
                                measurement='SOC',
                                tags=tags)

            if 'opt' in self.world.simulation_id:
                tempDF = pd.DataFrame(storage.profits, index=self.index, columns=['Profits_opt']).astype('float32')
            elif 'base' in self.world.simulation_id:
                tempDF = pd.DataFrame(storage.profits, index=self.index, columns=['Profits_base']).astype('float32')
            else:
                tempDF = pd.DataFrame(storage.profits, index=self.index, columns=['Profits']).astype('float32')

            self.writeDataFrame(dataframe=tempDF,
                                measurement='Profits',
                                tags=tags)

            if storage.bidding_type == 'opt':
                tempDF = pd.DataFrame(storage.opt_profits, index=self.index, columns=['Profits_max']).astype('float32')
                self.writeDataFrame(dataframe=tempDF,
                                     measurement='Profits',
                                    tags=tags)
                                        
    def delete_simulation(self, simID):
        self.dfClient.delete_series(tags={'simulationID': simID})
        print(simID, 'deleted')


    def delete_multiple_simulations(self, simIDs):
        reply = input(f'Are you sure you want to delete {str(simIDs)} ???')
        if reply.lower() in ['yes', 'y']:
            for simID in simIDs:
                self.delete_simulation(simID)
        else:
            print('!!! Ok, NOT deleted !!!')

    def delete_database(self, database_name):
        check = input('Are you sure you want to delete ' + database_name + ' ??? Type database name to confirm: ')
        if check == database_name:
            check = input('You are about to delete ' + database_name + ' ??? Are you absolutely sure??? [yes/no]: ')
            if check.lower() in ['yes', 'y']:
                self.dfClient.drop_database(database_name)
                print(database_name, 'database deleted')
            else:
                print('Ok, not deleted')

        else:
            print('!!! Wrong name entered !!!')

