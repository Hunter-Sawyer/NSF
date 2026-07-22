import string
import os,sys
from turtle import pd
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import activity_gen_assign_populations


if __name__ == "__main__":
    # Example usage
    #output_path = "./traci_test_PA.xml"
    #activity_gen_assign_populations.assign_pop_to_streets_with_from_census(output_path=output_path)
    #output_path = "./chargers_additional.xml"
    #activity_gen_assign_populations.assign_charging_stations_to_streets()

    activity_gen_assign_populations.assign_vehicles_new_vtype()

