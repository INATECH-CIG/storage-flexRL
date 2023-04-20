 # -*- coding: utf-8 -*-
"""
Created on Sun Apr  19 16:06:57 2020

@author: intgridnb-02
"""
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from misc import initializer
from bid import Bid

class OptStorage():
    
    @initializer
    def __init__(self,
                 agent=None,
                 name='Storage_1',
                 technology='PSPP',
                 bidding_type='opt',
                 min_soc=1,
                 max_soc=1000,
                 max_power_ch=100,
                 max_power_dis=100,
                 efficiency_ch=0.8,
                 efficiency_dis=0.9,
                 ramp_up=100,
                 ramp_down=100,
                 variable_cost_ch=0.28,
                 variable_cost_dis=0.28,
                 natural_inflow = 1.8, # [MWh/qh]
                 company = 'UNIPER',
                 world = None,
                 **kwargs):

        # Unit status parameters
        self.foresight = int(48/self.world.dt)
        self.crm_timestep = self.world.crm_timestep

    def reset(self):
        self.total_capacity = [0. for _ in self.world.snapshots]

        self.soc = [0. for _ in range(len(self.world.snapshots)+1)]
        self.soc[0] = 0.5*self.max_soc
        self.soc[-1] = self.min_soc

        self.energy_cost = [0. for _ in range(len(self.world.snapshots)+1)]

        self.bids_supply = {n:(0.,0.) for n in self.world.snapshots}
        self.bids_demand = {n:(0.,0.) for n in self.world.snapshots}

        self.sent_bids=[]

        self.rewards = [0. for _ in self.world.snapshots]
        self.profits = [0. for _ in self.world.snapshots]

        self.opt_profits = [0. for _ in self.world.snapshots]
        pfc = pd.read_csv(f'input/{self.world.scenario}/mcp.csv')
        self.pfc = pfc['Price'].tolist()
        self.pfc = self.world.pfc[:]

        self.optimal_strategy(timestep=0, foresight=len(self.world.snapshots)-1, mode='reset')

    def formulate_bids(self, t, market="EOM"):
        bids = []
        
        if market == "EOM":
            bids.extend(self.calculate_bids_eom())
                        
        return bids
      

    def step(self):
        t = self.world.currstep
        conf_bid_supply, conf_bid_demand = 0., 0.
            
        for bid in self.sent_bids:
            if 'supplyEOM' in bid.ID:
                conf_bid_supply = bid.confirmedAmount
                self.bids_supply[t] = (bid.confirmedAmount, bid.price)
            if 'demandEOM' in bid.ID:
                conf_bid_demand = bid.confirmedAmount
                self.bids_demand[t] = (bid.confirmedAmount, bid.price)

        self.total_capacity[t] = conf_bid_supply-conf_bid_demand

        self.soc[t+1] = self.soc[t] + (conf_bid_demand*self.efficiency_ch - conf_bid_supply/self.efficiency_dis)*self.world.dt
        self.soc[t+1] = max(self.soc[t+1], self.min_soc)

        if self.soc[t+1] >= self.min_soc+self.world.minBidEOM:
            self.energy_cost[t+1] = (self.energy_cost[t]*self.soc[t] - self.total_capacity[t]*self.world.mcp[t]*self.world.dt)/self.soc[t+1]
        else:
            self.energy_cost[t+1] = 0.

        profit = (conf_bid_supply-conf_bid_demand)*self.world.mcp[t]*self.world.dt
        profit -= (conf_bid_supply*self.variable_cost_dis + conf_bid_demand*self.variable_cost_ch)

        scaling = 0.1/self.max_power_ch

        self.rewards[t] = profit*scaling
        self.profits[t] = profit

        self.sent_bids=[]

        
    def feedback(self, bid):
        if bid.status in ["Confirmed", "PartiallyConfirmed"]:
            t = self.world.currstep

            if 'CRMPosDem' in bid.ID:
                self.confQtyCRM_pos.update(
                    {t+_: bid.confirmedAmount for _ in range(self.crm_timestep)})

            if 'CRMNegDem' in bid.ID:
                self.confQtyCRM_neg.update(
                    {t+_: bid.confirmedAmount for _ in range(self.crm_timestep)})
            
        self.sent_bids.append(bid)


    def calculate_bids_eom(self):
        t = self.world.currstep

        bid_quantity_demand, bid_quantity_supply, average_profit = self.optimal_strategy(timestep=t, foresight=self.foresight)
        bids = []

        if bid_quantity_supply >= self.world.minBidEOM:
            bids.append(
                Bid(
                    issuer=self,
                    ID=f"{self.name}_supplyEOM",
                    price=self.pfc[t]-average_profit,
                    amount=bid_quantity_supply,
                    status="Sent",
                    bidType="Supply",
                    node=self.node,
                )
            )

        if bid_quantity_demand >= self.world.minBidEOM:
            bids.append(
                Bid(
                    issuer=self,
                    ID=f"{self.name}_demandEOM",
                    price=self.pfc[t]+average_profit,
                    amount=bid_quantity_demand,
                    status="Sent",
                    bidType="Demand",
                    node=self.node,
                )
            )

        return bids


    def optimal_strategy(self, timestep, foresight=48, mode='bid'):
        if timestep+foresight < len(self.world.snapshots):
            local_pfc = self.pfc[timestep:timestep+foresight]
        else:
            local_pfc = self.pfc[timestep:]
            local_pfc.extend(self.pfc[:foresight-len(local_pfc)])
        
        #create model
        model = pyo.ConcreteModel()

        #define indices
        model.t = pyo.RangeSet(0, foresight-1)

        #define variables
        model.p_charge = pyo.Var(model.t, domain=pyo.NonNegativeReals, bounds = (0.0, self.max_power_ch))
        model.p_discharge = pyo.Var(model.t, domain=pyo.NonNegativeReals, bounds = (0.0, self.max_power_dis))
        model.soc = pyo.Var(model.t, domain=pyo.NonNegativeReals, bounds = (self.min_soc, self.max_soc))
        model.profit = pyo.Var(model.t, domain=pyo.Reals)

        #objective
        def rule_objective(model):
            return pyo.quicksum(model.profit[t] for t in model.t)

        model.obj = pyo.Objective(rule=rule_objective, sense=pyo.maximize)

    #constraints    
        def soc_rule(model, t):
            if t == 0:
                return model.soc[t] == self.soc[timestep] + model.p_charge[t]*self.efficiency_ch - model.p_discharge[t]/self.efficiency_dis
            else:
                return model.soc[t] == model.soc[t-1] + model.p_charge[t]*self.efficiency_ch - model.p_discharge[t]/self.efficiency_dis

        model.soc_rule = pyo.Constraint(model.t, rule=soc_rule)

        def final_soc_rule(model):
            return model.soc[foresight-1] == self.soc[timestep]

        model.final_soc_rule = pyo.Constraint(rule=final_soc_rule)

        def profit_rule(model, t):
            expr = model.profit[t] == (model.p_discharge[t]-model.p_charge[t])*local_pfc[t]*self.world.dt -\
                        (model.p_discharge[t]*self.variable_cost_dis +
                 model.p_charge[t]*self.variable_cost_ch)
            return expr

        model.profit_rule = pyo.Constraint(model.t, rule=profit_rule)

        # solve model
        opt = SolverFactory("gurobi_direct")
        opt.solve(model)

        if mode=='bid':
            total_dis = 0
            for t in range(foresight):
                total_dis += model.p_discharge[t].value

            p_ch = model.p_charge[0].value
            p_dis = model.p_discharge[0].value
            average_profit = 0 if total_dis == 0 else pyo.value(model.obj)/total_dis

            return p_ch, p_dis, average_profit

        elif mode=='reset':
            for t in range(foresight):
                self.opt_profits[t] = model.profit[t].value
