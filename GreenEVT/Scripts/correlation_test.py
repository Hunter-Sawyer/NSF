import string
import os,sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import sumolib
import logging
import numpy as np
import sqlite3
from xml.dom import minidom
import os
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

def test_correlation(output_path = r"C:\Users\Hunter\Downloads\RA\GreenEVT\data\Edge_detector_1.xml", road_activity = r".\matched_edges.txt"):
    #read sumo outputs
    sumo_outputs = minidom.parse(output_path)
    edges = sumo_outputs.getElementsByTagName("edge")
    #print(f"Total edges in SUMO output: {len(edges)}")

    edge_dict = {}
    for edge in edges:
        edge_id = edge.getAttribute("id")
        edge_left = edge.getAttribute("left")
        edge_dict[edge_id] = edge_left

    #read road activity file
    road_activity_dict = {}
    with open(road_activity, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                aadt = parts[0]
                edge_id = parts[1]
                road_activity_dict[edge_id] = aadt

    #compare
    sumo_output = []
    RWD = []
    for edge_id, activity in road_activity_dict.items():
        if edge_id in edge_dict:
            sumo_left = edge_dict[edge_id]
            #print(f"Edge ID: {edge_id}, AADT: {activity}, Left the street: {sumo_left}")
            sumo_output.append(float(sumo_left))
            RWD.append(float(activity))
        else:
            print(f"Edge ID: {edge_id} from road activity file not found in SUMO outputs.")

    sumo_output = sumo_output

    correlation = np.corrcoef(sumo_output, RWD)[0,1]
    #print(f"Correlation coefficient between SUMO left street and AADT: {correlation}")
    return sumo_output, RWD, correlation

def count_jobs(stats_file = r"..\data\statistics_TAZ_Jobs_test.xml"):
    stats_dom = minidom.parse(stats_file)
    tazs = stats_dom.getElementsByTagName("street")
    total_jobs = 0
    for taz in tazs:
        jobs = taz.getAttribute("workPosition")
        total_jobs += int(jobs)
    print(f"Total jobs across all streets: {total_jobs}")

def plot_correlation(sumo_output, RWD, correlation):


    #select a subset for clearer plotting
    #indices = np.linspace(0, len(sumo_output)-1, num=60, dtype=int)
    #sumo_output = [sumo_output[i] for i in indices]
    #RWD = [RWD[i] for i in indices]
    
    #NORMALIZE DATA
    #sumo_output = (sumo_output - np.min(sumo_output)) / (np.max(sumo_output) - np.min(sumo_output))
    #RWD = (RWD - np.min(RWD)) / (np.max(RWD) - np.min(RWD))

    fig,ax1 = plt.subplots()

    ax1.plot(range(len(RWD)),RWD, 'r-o', label='Road Activity Data (AADT)')
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Road Activity Data (AADT)", color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(range(len(sumo_output)),sumo_output, 'b-o', label='SUMO Left the Street Count')
    ax2.set_ylabel("SUMO Left the Street Count", color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title(f"Correlation between Road Activity Data and SUMO Outputs: {correlation:.2f}")
    plt.grid(True)
    plt.show()



    rho, pval = spearmanr(sumo_output, RWD)
    print(f"Spearman correlation: {rho}, p-value: {pval}")

    plt.scatter(RWD, sumo_output)
    plt.xlim(0,40000)
    plt.ylim(0,80)
    plt.xlabel("Road Activity Data (AADT)")
    plt.ylabel("SUMO Left the Street Count")
    plt.title("Scatter Plot of Road Activity Data vs SUMO Outputs")
    plt.grid(True)
    plt.show()

#sumo_output, RWD, correlation = correlation_test()
#plot_correlation(sumo_output, RWD, correlation)

#count_jobs()