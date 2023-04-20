 # -*- coding: utf-8 -*-
"""
Created on Sun Apr  19 16:06:57 2020

@author: intgridnb-02
"""
import numpy as np

from misc import initializer
from bid import Bid

class Storage():
    
    @initializer
    def __init__(self,
                 agent=None,
                 name='Storage_1',
                 technology='PSPP',
                 bidding_type='rule_based',
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
        self.foresight = int(12/self.world.dt)
        self.crm_timestep = self.world.crm_timestep

    def reset(self):
        self.total_capacity = [0. for _ in self.world.snapshots]

        self.soc = [0. for _ in range(len(self.world.snapshots)+1)]
        self.soc[0] = 0.5*self.max_soc
        self.soc[-1] = self.min_soc

        self.energy_cost = [0. for _ in range(len(self.world.snapshots)+1)]

        self.bids_supply = {n:(0.,0.) for n in self.world.snapshots}
        self.bids_demand = {n:(0.,0.) for n in self.world.snapshots}
        self.confQtyCRM_neg = {n:0 for n in self.world.snapshots}
        self.confQtyCRM_pos = {n:0 for n in self.world.snapshots}

        self.sent_bids=[]

        self.rewards = [0. for _ in self.world.snapshots]
        self.profits = [0. for _ in self.world.snapshots]


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

        self.energy_cost[t+1] = max(self.energy_cost[t+1], -100)
        self.energy_cost[t+1] = min(self.energy_cost[t+1], 100)

        profit = (conf_bid_supply-conf_bid_demand)*self.world.mcp[t]*self.world.dt
        profit -= (conf_bid_supply*self.variable_cost_dis + conf_bid_demand*self.variable_cost_ch)

        scaling = 0.1/self.max_power_ch

        self.rewards[t] = profit*scaling
        self.profits[t] = profit

        self.sent_bids=[]

        
    def feedback(self, bid):
        if bid.status in ['Confirmed', 'PartiallyConfirmed']:
            t = self.world.currstep

            if 'CRMPosDem' in bid.ID:
                self.confQtyCRM_pos.update(
                    {t+_: bid.confirmedAmount for _ in range(self.crm_timestep)})

            if 'CRMNegDem' in bid.ID:
                self.confQtyCRM_neg.update(
                    {t+_: bid.confirmedAmount for _ in range(self.crm_timestep)})

        self.sent_bids.append(bid)


    def formulate_bids(self, t, market="EOM"):
        bids = []
        
        if market == "EOM":
            bids.extend(self.calculate_bids_eom(t))
            
        elif market == "posCRMDemand":
            bids.extend(self.calculatingBidsSTO_CRM_pos(t))

        elif market == "negCRMDemand":
            bids.extend(self.calculatingBidsSTO_CRM_neg(t))
            
        return bids
      

    def calculate_bids_eom(self, t, passedSOC = None):
        soc = self.soc[t] if passedSOC is None else passedSOC
        bidsEOM = []
        bidPrice_supply, bidPrice_demand = 0, 0

        if t >= len(self.world.snapshots):
            t -= len(self.world.snapshots)

        if t-self.foresight < 0:
            averagePrice = np.mean(self.world.pfc[t-self.foresight:] + self.world.pfc[:t+self.foresight])

        elif t+self.foresight > len(self.world.snapshots):
            averagePrice = np.mean(self.world.pfc[t-self.foresight:] + self.world.pfc[:t+self.foresight-len(self.world.snapshots)])

        else:
            averagePrice = np.mean(self.world.pfc[t-self.foresight:t+self.foresight])


        if self.world.pfc[t] >= averagePrice/self.efficiency_dis:
            bid_quantity_supply = min(((soc-self.min_soc)/self.world.dt - self.confQtyCRM_pos[t])*self.efficiency_dis,
                                     self.max_power_dis)

            bidPrice_supply = averagePrice

            if bid_quantity_supply >= self.world.minBidEOM*self.efficiency_dis:
                bidsEOM.append(
                    Bid(
                        issuer=self,
                        ID=f"{self.name}_supplyEOM",
                        price=bidPrice_supply,
                        amount=bid_quantity_supply,
                        status="Sent",
                        bidType="Supply",
                        node=self.node,
                    )
                )

        elif self.world.pfc[t] <= averagePrice*self.efficiency_ch:
            bid_quantity_demand = min(((self.max_soc-soc)/self.world.dt - self.confQtyCRM_neg[t])/self.efficiency_ch,
                                     self.max_power_ch)

            bidPrice_demand = averagePrice

            if bid_quantity_demand >= self.world.minBidEOM:
                bidsEOM.append(
                    Bid(
                        issuer=self,
                        ID=f"{self.name}_demandEOM",
                        price=bidPrice_demand,
                        amount=bid_quantity_demand,
                        status="Sent",
                        bidType="Demand",
                        node=self.node,
                    )
                )

        return bidsEOM


    def calculatingBidPricesSTO_CRM(self, t):
        fl = int(4 / self.world.dt)
        theoreticalSOC = self.soc[t]
        theoreticalRevenue = []

        for tick in range(t, t + fl):
            BidSTO_EOM = self.calculate_bids_eom(tick, theoreticalSOC)

            if len(BidSTO_EOM) == 0:
                continue

            BidSTO_EOM = BidSTO_EOM[0]
            if BidSTO_EOM.bidType == 'Demand':
                theoreticalSOC += BidSTO_EOM.amount * self.efficiency_ch * self.world.dt
                theoreticalRevenue.append(- self.world.pfc[t] * BidSTO_EOM.amount * self.world.dt)

            elif BidSTO_EOM.bidType == 'Supply':
                theoreticalSOC -= BidSTO_EOM.amount / self.efficiency_dis * self.world.dt
                theoreticalRevenue.append(self.world.pfc[t] * BidSTO_EOM.amount * self.world.dt)

        capacityPrice = abs(sum(theoreticalRevenue))
        energyPrice = -self.energy_cost[self.world.currstep] / self.soc[t]

        return capacityPrice, energyPrice


    def calculatingBidsSTO_CRM_pos(self, t):
        bidsCRM = []

        availablePower_BP_pos = min(max((self.soc[t] - self.min_soc) * self.efficiency_dis / self.world.dt, 0),
                                    self.max_power_dis)

        if availablePower_BP_pos >= self.world.minBidCRM:
            bidQuantityBPM_pos = availablePower_BP_pos
            capacityPrice, energyPrice = self.calculatingBidPricesSTO_CRM(t)

            bidsCRM.append(
                Bid(
                    issuer=self,
                    ID=f"{self.name}_CRMPosDem",
                    price=capacityPrice,
                    amount=bidQuantityBPM_pos,
                    energyPrice=energyPrice,
                    status="Sent",
                    bidType="Supply",
                )
            )

        else:
            bidsCRM.append(
                Bid(
                    issuer=self,
                    ID=f"{self.name}_CRMPosDem",
                    price=0,
                    amount=0,
                    energyPrice=0,
                    status="Sent",
                    bidType="Supply",
                )
            )
                    
        return bidsCRM
    

    def calculatingBidsSTO_CRM_neg(self, t):
        bidsCRM = []

        availablePower_BP_neg = min(max((self.max_soc - abs(self.soc[t])) / self.efficiency_ch / self.world.dt, 0),
                                    self.max_power_ch)

        if availablePower_BP_neg >= self.world.minBidCRM:
            
            bidQtyCRM_neg = availablePower_BP_neg

            bidsCRM.append(
                Bid(
                    issuer=self,
                    ID=f"{self.name}_CRMNegDem",
                    price=0,
                    amount=bidQtyCRM_neg,
                    energyPrice=0,
                    status="Sent",
                    bidType="Supply",
                )
            )

        else:
            bidsCRM.append(
                Bid(
                    issuer=self,
                    ID=f"{self.name}_CRMNegDem",
                    price=0,
                    amount=0,
                    energyPrice=0,
                    status="Sent",
                    bidType="Supply",
                )
            )
                    
        return bidsCRM

