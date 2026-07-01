import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

def main():
    traci_cmd = ["sumo",
                "--net-file","..\\data\\Palo Alto\\PA.network.net.xml",
                "--route-files", ".\\test_PA_rou.xml",
                "--additional-files", ".\\chargers_additional.xml",
                "--chargingstations-output", ".\\charging_stations_output.xml"]
    traci.start(traci_cmd)
    for step in range(1000):
        print(f"Simulation step: {step}")
        traci.simulationStep()



if __name__ == "__main__":
    main()