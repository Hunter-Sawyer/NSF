import string
import os,sys
from turtle import pd
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import sumolib
import logging
import numpy as np
import sqlite3
from xml.dom import minidom
import os
from tqdm import tqdm
import random
import geopandas as gpd


#This file creates the XML for the statistics file, including all default params, not including assigning pops to streers
def create_stat_XML(inhabitants="1000",households="500",childrenAgeLimit="19",retirementAgeLimit="66",carRate="0.58",unemploymentRate="0.05",footDistanceLimit="250",incomingTraffic="200",outgoingTraffic="50",laborDemand="1.05"):

    #creates XML objext (minidom)
    doc = minidom.Document()
    root = doc.documentElement

    #creates city element as main element in XML tree, top level
    doc.appendChild(doc.createElement("city"))

    #creates general element and adds attributes
    general = doc.createElement("general")
    general.setAttribute("inhabitants", inhabitants)
    general.setAttribute("households", households)
    general.setAttribute("childrenAgeLimit", childrenAgeLimit)
    general.setAttribute("retirementAgeLimit", retirementAgeLimit)
    general.setAttribute("carRate", carRate)
    general.setAttribute("unemploymentRate", unemploymentRate)
    general.setAttribute("footDistanceLimit", footDistanceLimit)
    general.setAttribute("incomingTraffic", incomingTraffic)
    general.setAttribute("outgoingTraffic", outgoingTraffic)
    general.setAttribute("laborDemand", laborDemand)

    doc.documentElement.appendChild(general)

    #creates parameters element and adds attributes
    parameters = doc.createElement("parameters")
    parameters.setAttribute("carPreference","0.50")
    parameters.setAttribute("meanTimePerKmInCity","6")
    parameters.setAttribute("freeTimeActivityRate","0.15")
    parameters.setAttribute("uniformRandomTraffic","0.20")
    parameters.setAttribute("departureVariation","300")

    doc.documentElement.appendChild(parameters)

    #creates population elements and adds brackets
    population = doc.createElement("population")
    bracket = doc.createElement("bracket")
    bracket.setAttribute("beginAge", "0")
    bracket.setAttribute("endAge", "30")
    bracket.setAttribute("peopleNbr","30")
    population.appendChild(bracket)

    bracket = doc.createElement("bracket")
    bracket.setAttribute("beginAge", "30")
    bracket.setAttribute("endAge", "60")
    bracket.setAttribute("peopleNbr","40")
    population.appendChild(bracket)

    bracket = doc.createElement("bracket")
    bracket.setAttribute("beginAge", "60")
    bracket.setAttribute("endAge", "90")
    bracket.setAttribute("peopleNbr","30")

    population.appendChild(bracket)

    doc.documentElement.appendChild(population)

    workHours = doc.createElement("workHours")

    opening = doc.createElement("opening")
    opening.setAttribute("hour","30600")
    opening.setAttribute("proportion","0.30")
    workHours.appendChild(opening)

    opening = doc.createElement("opening")
    opening.setAttribute("hour","32400")
    opening.setAttribute("proportion","0.70")
    workHours.appendChild(opening)

    closing = doc.createElement("closing")
    closing.setAttribute("hour","43200")
    closing.setAttribute("proportion","0.20")
    workHours.appendChild(closing)

    closing = doc.createElement("closing")
    closing.setAttribute("hour","63000")
    closing.setAttribute("proportion","0.20")
    workHours.appendChild(closing)

    closing = doc.createElement("closing")
    closing.setAttribute("hour","64800")
    closing.setAttribute("proportion","0.60")
    workHours.appendChild(closing)   

    doc.documentElement.appendChild(workHours)

    doc.documentElement.appendChild(doc.createElement("streets"))

    return doc

#shape="53509.41,17117.64 53487.55,17053.94 53485.57,17042.88 53485.62,17039.16 53494.05,17035.34 53550.35,17022.20"
#assigns populations to streets based on taz pop
def assign_pop_to_street(database_path = "../data/UDS.db", network_path = "../data/sumo_network/greensboro.net.xml",seed = 98,output_path = "..\data\genetic_algorithm_statistics.xml"):
    doc = create_stat_XML()
    #print("Created statistics XML")

    conn = sqlite3.connect(database_path)

    NetDom = minidom.parse(network_path)
    root = doc.documentElement
    

    #print("Grabbing TAZ data from database")
    Tazs = grab_taz(conn,network_path)
    
    taz_dict = {k: [] for k in Tazs}
    taz_pop_spread = {k : k[3] for k in Tazs}

    streets_node = root.getElementsByTagName("streets")[0]
    edges = list(NetDom.getElementsByTagName("edge"))
    print(f"Found {len(edges)} edges in the network")
    #print("starting assignments")

    for edge in edges:
        if edge.getAttribute("function") != "internal" and edge.getAttribute("shape") != '':   
         
            shape = edge.getAttribute("shape")
            #print(f"Original shape string: {shape}")
            #print(f"edge id: {edge.getAttribute('id')}")
            shape = shape.split(" ")
            for i in range(len(shape)):
                shape[i] = shape[i].split(",")
                shape[i] = (float(shape[i][0]),float(shape[i][1]))
            #print(f"Processing edge {edge.getAttribute('id')} with shape {shape}")

            
            
            min_taz = Tazs[0],sumolib.geomhelper.distancePointToPolygon((Tazs[0][1],Tazs[0][2]),shape)
            for taz in Tazs:
                dist = sumolib.geomhelper.distancePointToPolygon((taz[1],taz[2]),shape)
                if dist < min_taz[1]:
                    min_taz = taz,dist
            taz_dict[min_taz[0]].append(edge.getAttribute("id"))
            
        else:
            continue

    for taz in Tazs:
        taz_pop_spread[taz] = taz[3] / len(taz_dict[taz])
    
    
    num_internal = sum(1 for edge in edges if edge.getAttribute("function") == "internal" or edge.getAttribute("shape") == '')
    per_job = 154000 / (len(edges)-num_internal)
    job_assignment = {}
    print(f"Average jobs per street (excluding internal/no-shape edges): {per_job}")
    print(f"total non-internal edges: {len(edges)-num_internal}")
    print(f"internal/no-shape edges: {num_internal}")
    random.seed(seed)
    for edge in edges:
        if edge.getAttribute("function") != "internal" and edge.getAttribute("shape") != '':
            shape = edge.getAttribute("shape")
            shape = shape.split(" ")
            for i in range(len(shape)):
                shape[i] = shape[i].split(",")
                shape[i] = (float(shape[i][0]),float(shape[i][1]))

            min_taz = Tazs[0],sumolib.geomhelper.distancePointToPolygon((Tazs[0][1],Tazs[0][2]),shape)
            for taz in Tazs:
                dist = sumolib.geomhelper.distancePointToPolygon((taz[1],taz[2]),shape)
                if dist < min_taz[1]:
                    min_taz = taz,dist

            jobs = max(0,int(random.gauss(per_job,per_job/2)))
            street = doc.createElement("street")
            street.setAttribute("edge",str(edge.getAttribute("id")))
            street.setAttribute("population",str(taz_pop_spread[min_taz[0]]))
            street.setAttribute("workPosition",f"{jobs}")
            streets_node.appendChild(street)
        else:
            #jobs = max(0,int(random.gauss(per_job,.5)))
            street = doc.createElement("street")
            street.setAttribute("edge",str(edge.getAttribute("id")))
            street.setAttribute("population","0")
            street.setAttribute("workPosition","0")
            streets_node.appendChild(street)
    with open(output_path, "w") as f:
        doc.writexml(f, indent="  ", addindent="  ", newl="\n")

    pass

#returns list of taz ids,x,y with x and y being relative to sumo net
def grab_taz(conn,network_path = "../data/sumo_network/greensboro.net.xml"):
    c = conn.cursor()

    net = sumolib.net.readNet(network_path)

    c.execute("SELECT ogc_fid,x,y,population from taz")
    data = c.fetchall()


    for i in range(len(data)):
        x,y =net.convertLonLat2XY(float(data[i][1]),float(data[i][2]))
        #print(f"TAZ {data[i][0]} original coords: ({data[i][1]}, {data[i][2]}) converted to ({x}, {y})")
        data[i] = data[i][0],x,y,data[i][3]
        
    #print(f"taz:{type(data[i])}")
    #TAZ = [data]
    #print(f"Found {TAZ} TAZs")
    #print(f"TAZ {type(TAZ)}")
    #print(f"TAZ {TAZ[1]}")
    #print(f"TAZ {type(TAZ)}")
    print(f"type of data: {type(data)}")
    
    #print(type(data))
    #print(data)
    return data

def get_junctions(network_path = "../data/sumo_network/greensboro.net.xml"):
    NetDom = minidom.parse(network_path)
    junctions = list(NetDom.getElementsByTagName("junction"))
    junction_dict = {}
    print("Junctions found in network:")
    for junction in junctions:
        junction_id = junction.getAttribute("id")
        #print(f"Junction ID: {junction_id}")
        x = float(junction.getAttribute("x"))
        #print(f"X Coordinate: {x}")
        y = float(junction.getAttribute("y"))
        #print(f"Y Coordinate: {y}")
        junction_dict[junction_id] = (x, y)
    return junction_dict

def assign_pop_to_street_without_jobs_init(
        database_path = "../data/UDS.db", 
        network_path = "../data/sumo_network/greensboro.net.xml",
        seed=98,
        output_path = "../genetic_alg/static_files/pop_file.xml",inhabitants = 10000):
    doc = create_stat_XML(inhabitants=str(inhabitants))
    print("Created statistics XML")

    conn = sqlite3.connect(database_path)

    NetDom = minidom.parse(network_path)
    root = doc.documentElement

    print("Grabbing TAZ data from database")
    Tazs = grab_taz(conn,network_path)
    
    taz_dict = {k: [] for k in Tazs}
    taz_pop_spread = {k : k[3] for k in Tazs}

    streets_node = root.getElementsByTagName("streets")[0]
    edges = list(NetDom.getElementsByTagName("edge"))
    print(f"Found {len(edges)} edges in the network")
    print("starting assignments")

    for edge in tqdm(edges,desc="Assigning populations to streets"):
        if edge.getAttribute("function") != "internal" and edge.getAttribute("shape") != '':   
         
            shape = edge.getAttribute("shape")
            shape = shape.split(" ")
            for i in range(len(shape)):
                shape[i] = shape[i].split(",")
                shape[i] = (float(shape[i][0]),float(shape[i][1]))

            min_taz = Tazs[0],sumolib.geomhelper.distancePointToPolygon((Tazs[0][1],Tazs[0][2]),shape)
            for taz in Tazs:
                dist = sumolib.geomhelper.distancePointToPolygon((taz[1],taz[2]),shape)
                if dist < min_taz[1]:
                    min_taz = taz,dist
            taz_dict[min_taz[0]].append(edge.getAttribute("id"))
            
        else:
            continue

    for taz in Tazs:
        taz_pop_spread[taz] = taz[3] / len(taz_dict[taz])

    real_edges = []
    for edge in edges:
        if edge.getAttribute("function") != "internal" and edge.getAttribute("shape") != '':
            shape = edge.getAttribute("shape")
            shape = shape.split(" ")
            for i in range(len(shape)):
                shape[i] = shape[i].split(",")
                shape[i] = (float(shape[i][0]),float(shape[i][1]))

            min_taz = Tazs[0],sumolib.geomhelper.distancePointToPolygon((Tazs[0][1],Tazs[0][2]),shape)
            for taz in Tazs:
                dist = sumolib.geomhelper.distancePointToPolygon((taz[1],taz[2]),shape)
                if dist < min_taz[1]:
                    min_taz = taz,dist

            jobs = max(0,int(random.randint(0,10)))
            street = doc.createElement("street")
            street.setAttribute("edge",str(edge.getAttribute("id")))
            street.setAttribute("population",str(taz_pop_spread[min_taz[0]]))
            #street.setAttribute("workPosition",f"{jobs}")
            streets_node.appendChild(street)
            real_edges.append(edge.getAttribute("id"))

            #job_assignment[str(edge.getAttribute("id"))] = jobs
        else:
            #jobs = max(0,int(random.gauss(per_job,.5)))
            street = doc.createElement("street")
            street.setAttribute("edge",str(edge.getAttribute("id")))
            street.setAttribute("population","0")
            #street.setAttribute("workPosition","0")
            streets_node.appendChild(street)
    print(f"writing to file{output_path}")
    with open(output_path, "w") as f:
        doc.writexml(f, indent="  ", addindent="  ", newl="\n")

    with open("../genetic_alg/static_files/real_edges.txt","w") as f:
        for edge_id in real_edges:
            f.write(f"{edge_id}\n")
    return 

def assign_jobs(output_file_name,job_assignments,real_edges,population_file_template = "../genetic_alg/static_files/pop_file.xml"):
    doc = minidom.parse(population_file_template)
    root = doc.documentElement
    streets_node = root.getElementsByTagName("streets")[0]
    edges = list(streets_node.getElementsByTagName("street"))
    print(f"Found {len(edges)} edges in the population file")

    for edge in edges:
        edge_id = edge.getAttribute("edge")
        if edge_id in real_edges:
            index = real_edges.index(edge_id)
            jobs = job_assignments[index]
            edge.setAttribute("workPosition",f"{jobs}")
        else:
            edge.setAttribute("workPosition","0")
    with open(output_file_name, "w") as f:
        doc.writexml(f)
    return
    
def assign_pop_to_streets_with_from_census(block_groups_path = "..\\..\\tl_2019_06_bg\\tl_2019_06_bg.shp", 
associated_pop_path = "..\\..\\tl_2019_06_bg\\block_groups_pop\\ACSDT5Y2020.B01003-Data.csv",
network_path = "..\\data\\Palo Alto\\PA.network.net.xml",
output_path = "../genetic_alg/static_files/pop_file.xml"):
    doc = create_stat_XML()

    block_groups = gpd.read_file(block_groups_path)
    pop_data = pd.read_csv(associated_pop_path)

    block_groups_df = create_pop_lookup(block_groups_path, associated_pop_path)

    NetDom = minidom.parse(network_path)
    root = doc.documentElement

    streets_node = root.getElementsByTagName("streets")[0]
    edges = list(NetDom.getElementsByTagName("edge"))

    for taz in block_groups_df.itertuples():
        taz_population = taz.population
        taz_shape = taz.geometry
        edges_within_taz = []
        for edge in edges:
            if edge.getAttribute("function") != "internal" and edge.getAttribute("shape") != '':
                shape = edge.getAttribute("shape")
                shape = shape.split(" ")
                for i in range(len(shape)):
                    shape[i] = shape[i].split(",")
                    shape[i] = (float(shape[i][0]), float(shape[i][1]))

                dist = sumolib.geomhelper.distancePointToPolygon((taz_shape.centroid.x, taz_shape.centroid.y), shape)
                if dist <= 1:  # Adjust the threshold as needed
                    edges_within_taz.append(edge.getAttribute("id"))
        
        if edges_within_taz:
            pop_per_edge = taz_population / len(edges_within_taz)
            for edge_id in edges_within_taz:
                street = doc.createElement("street")
                street.setAttribute("edge", str(edge_id))
                street.setAttribute("population", str(pop_per_edge))
                street.setAttribute("workPosition", "0")  # Default work position
                streets_node.appendChild(street)





def create_pop_lookup(block_groups_path = "..\\..\\tl_2019_06_bg\\tl_2019_06_bg.shp", 
associated_pop_path = "..\\..\\tl_2019_06_bg\\block_groups_pop\\ACSDT5Y2020.B01003-Data.csv"):
    block_groups = gpd.read_file(block_groups_path)
    pop_data = pd.read_csv(associated_pop_path)

    geo_id_list = pop_data['GEO_ID'].tolist()
    sublist = [s[-12:] for s in geo_id_list]

    # Create matching GEOID column in the CSV
    pop_data['GEOID'] = pop_data['GEO_ID'].str[-12:]

    # Create lookup dictionary
    pop_lookup = dict(
        zip(
            pop_data['GEOID'],
            pop_data['B01003_001E']
        )
    )

    block_groups['population'] = block_groups['GEOID'].map(pop_lookup)
    block_groups = block_groups.dropna(subset=['population'])  # Drop rows with NaN population values

    return block_groups


#assign_pop_to_street()

#junctions = get_junctions()
#TAZ = grab_taz(sqlite3.connect("../data/UDS.db"))




