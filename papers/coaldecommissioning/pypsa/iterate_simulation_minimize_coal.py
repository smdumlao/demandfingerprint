from __future__ import print_function, division, absolute_import
import pypsa
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sys import argv
import winsound

#--------- Directory management
cwd = os.getcwd()
csv_folder_name = os.path.join(os.getcwd(), 'csv_folder') + os.sep
LNG_limit = int(argv[1])

#-------------------- load and back up ------------------------#
network = pypsa.Network()
network.import_from_csv_folder(csv_folder_name)

#----------------------- co2 constraints---------------------- #

#lng limit
network.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=LNG_limit*1_000_000)

network.add("Carrier","gas",co2_emissions=1) # in t_CO2/MWht
network.generators.at['LNG', 'carrier'] = 'gas'

#--------- Set Coal Decommission Parameters --------------------#
network.generators.at['Coal', 'p_nom_min'] = 0
network.generators.at['Coal', 'p_nom_max'] = 7037

#-------------- Assign Year ------------------------------------#
syn_demand = pd.read_csv(csv_folder_name+'syn_demand.csv', index_col = 'datetime')
syn_demand.index = pd.to_datetime(syn_demand.index)

solar_p_max = pd.read_csv(csv_folder_name+'solar_weather_p_max_pu.csv', index_col='datetime')
solar_p_max.index = pd.to_datetime(solar_p_max.index)

#results
results_dir = os.path.join(os.getcwd(), 'results') + os.sep

for year in syn_demand.columns:
    # change the demand of Kyushu
    network.loads_t.p_set['Kyushu'] = syn_demand[year]
    print(network.loads_t.p_set.head(10))
    
    network.generators_t.p_max_pu['Solar'] = solar_p_max[year]
    print(network.generators_t.p_max_pu['Solar'].head(24))
    
    
    # save loc
    directory = results_dir+ f'{LNG_limit}_{year}' + os.sep
    if not os.path.exists(directory):
        os.makedirs(directory)

    #---------------------set capacity -----------------------------#
    pv_capacity=[]
    pv_capacity.extend(np.arange(0,1000,500))
    pv_capacity.extend(np.arange(1000,20001, 1000))
    
    print (pv_capacity)
    print (len(pv_capacity))

    for pv_cap in pv_capacity:
        network.generators.p_nom.at['Solar'] = pv_cap
        network.lopf(network.snapshots, solver_name='gurobi');
        
        #save network for processing later
        network_result = f'{directory}/network_{str(pv_cap).zfill(5)}'
        if not os.path.exists(network_result):
            os.makedirs(network_result)
        network.export_to_csv_folder(network_result)
            
        # total_power = network.generators_t.p.sum()
        print (pv_cap)
        print (network.generators.p_nom_opt)

winsound.Beep(8000,200)
winsound.Beep(8000,500)
winsound.Beep(8000,200)
winsound.Beep(8000,500)
winsound.Beep(8000,200)
winsound.Beep(8000,500)
