import string
import os,sys
#from turtle import pd
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
from pathlib import Path
import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString
from pyproj import CRS
from network_filter import passenger_connectivity

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parent


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
    parameters.setAttribute("freeTimeActivityRate","0.00")
    parameters.setAttribute("uniformRandomTraffic","0.00")
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
        output_path = "../genetic_alg/static_files/pop_file.xml",inhabitants = 10000
        ,incoming_percent = .15, outgoing_percent = .15,SDN2 = ""):
    doc = create_stat_XML(inhabitants=str(inhabitants),households = str(inhabitants/2),outgoingTraffic=str(int(inhabitants*outgoing_percent)),incomingTraffic=str(int(inhabitants*incoming_percent)))
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

    with open(f"../genetic_alg/{SDN2}static_files/real_edges.txt","w") as f:
        for edge_id in real_edges:
            f.write(f"{edge_id}\n")
    return 

def assign_jobs(output_file_name,job_assignments,real_edges,n_gates,IO_list = [200,150],population_file_template = "../genetic_alg/static_files/pop_file.xml",SDN2 = "",network_path = "../data/Palo_Alto/PA.network.net.xml"):
    doc = minidom.parse(population_file_template)
    root = doc.documentElement
    general = doc.getElementsByTagName("general")[0]
    general.setAttribute("incomingTraffic",f"{int(IO_list[0])}")
    general.setAttribute("outgoingTraffic",f"{int(IO_list[1])}")
    streets_node = root.getElementsByTagName("streets")[0]
    edges = list(streets_node.getElementsByTagName("street"))
    #print(f"Found {len(edges)} edges in the population file")

    edge_job_map = dict(zip(real_edges, job_assignments))

    #print("Setting work Positions")
    for edge in edges:
        edge_id = edge.getAttribute("edge")
        try:
            edge.setAttribute("workPosition",str(edge_job_map.get(edge_id, 0)))
        except KeyError:
            edge.setAttribute("workPosition","0")
    network = sumolib.net.readNet(network_path)
    connectivity = passenger_connectivity(network)

    with open("../genetic_alg/static_files/gates.txt", "r") as f:
        gates = [line.strip() for line in f.readlines() if line.strip()]

    city_gates_node = doc.createElement("cityGates")

    #print("Reading Gates")
    for idx, gate_edge_id in enumerate(gates):
        try:
            gate_edge = network.getEdge(gate_edge_id)
        except KeyError:
            print(f"Skipping gate edge missing from current network: {gate_edge_id}")
            continue

        has_incoming = bool(gate_edge.getIncoming())
        has_outgoing = bool(gate_edge.getOutgoing())

        # An incoming gate has no predecessor; an outgoing gate has no
        # successor. Do not assign traffic to edges that are not true gates.
        if has_incoming and has_outgoing:
            print(f"Skipping non-gate edge in gates.txt: {gate_edge_id}")
            continue

        if not has_incoming and gate_edge_id not in connectivity["from_hub"]:
            print(f"Skipping departure gate with no directed path from network hub: {gate_edge_id}")
            continue

        if not has_outgoing and gate_edge_id not in connectivity["to_hub"]:
            print(f"Skipping arrival gate with no directed path to network hub: {gate_edge_id}")
            continue

        # ActivityGen's incoming traffic ends at a city gate and outgoing
        # traffic starts at one.  Therefore the XML attributes are opposite
        # the topological test: no successor -> incoming, no predecessor ->
        # outgoing.
        incoming_val = n_gates[idx, 0] if not has_outgoing else 0
        outgoing_val = n_gates[idx, 1] if not has_incoming else 0

        gate_node = doc.createElement("entrance")
        gate_node.setAttribute("edge", gate_edge_id)
        gate_node.setAttribute("pos","0.00")
        gate_node.setAttribute("incoming",f"{incoming_val}")
        gate_node.setAttribute("outgoing",f"{outgoing_val}")

        city_gates_node.appendChild(gate_node)
        root.appendChild(city_gates_node)

    with open(output_file_name, "w") as f:
        doc.writexml(f)
    return
    
def assign_pop_to_streets_with_from_census(
    block_groups_path=None,
    associated_pop_path=None,
    network_path=None,
    output_path=None,
    population=1000,
    SDN2="",
):
    """Create an ActivityGen population file from census block groups.

    Omitted paths follow the repository layout. Explicit relative paths stay
    relative to the process working directory on both Windows and Linux.
    """
    block_groups_path = Path(block_groups_path or (
        REPOSITORY_DIR / "tl_2019_06_bg" / "tl_2019_06_bg.shp"
    )).expanduser()
    associated_pop_path = Path(associated_pop_path or (
        REPOSITORY_DIR
        / "tl_2019_06_bg"
        / "block_groups_pop"
        / "ACSDT5Y2020.B01003-Data.csv"
    )).expanduser()
    network_path = Path(network_path or (
        PROJECT_DIR / "data" / "Palo Alto" / "PA.network.net.xml"
    )).expanduser()
    output_path = Path(output_path or (
        PROJECT_DIR / "genetic_alg" / f"{SDN2}static_files" / "pop_file.xml"
    )).expanduser()

    if not isinstance(population, (int, float)) or population <= 0:
        raise ValueError("population must be a positive number")

    inhabitants = int(population)
    # Keep the existing household-to-inhabitant ratio while satisfying
    # ActivityGen's requirement that households be fewer than inhabitants.
    households = max(1, int(round(inhabitants * 0.5)))
    doc = create_stat_XML(
        inhabitants=str(inhabitants),
        households=str(households),
    )

    #block_groups = gpd.read_file(block_groups_path)
    #pop_data = pd.read_csv(associated_pop_path)

    block_groups_df = create_pop_lookup(
        str(block_groups_path),
        str(associated_pop_path),
    )
    print(f"Loaded {len(block_groups_df)} block groups with population data")
    print(f"CRS of block groups: {block_groups_df.crs}")

    net = sumolib.net.readNet(str(network_path))
    print(f"Loaded SUMO network with {len(net.getEdges())} edges")

    # try:
    proj_string = net.getLocationOffset()

    net_proj = net.getGeoProj()
    proj4_string = net_proj.srs

    if net_proj:
        print("SUMO projection found:")
        print(net_proj)

        block_groups_df = block_groups_df.to_crs(CRS.from_proj4(proj4_string))

        print("Reprojected TAZs to SUMO CRS")
    
    # except Exception as e:
    #     print("WARNING: Could not automatically project TAZs")
    #     print(e)
    #     print(
    #         "Make sure your block groups and SUMO network "
    #         "already use the same coordinate system."
    #     )

    # print(block_groups_df.geometry.iloc[5])
    # print(net.getEdges()[0].getShape()[0])
    print(net.getLocationOffset())

    ox, oy = net.getLocationOffset()
    block_groups_df['geometry'] = block_groups_df['geometry'].translate(xoff=ox, yoff=oy)
    # print(block_groups_df.geometry.iloc[5])
    print(net.getLocationOffset())
    print(net.getEdges()[0].getShape()[0])

    # exit()

    root = doc.documentElement
    streets_node = root.getElementsByTagName("streets")[0]

    # Pre-build edge geometries once
    print("Building edge geometries...")

    edge_geometries = []
    real_edges = []
    connected_edges = passenger_connectivity(net)["round_trip"]
    for edge in net.getEdges():
        # Skip internal/junction edges
        if edge.isSpecial():
            #print(f"Skipping special edge {edge.getID()}")
            continue

        if edge.getID() not in connected_edges:
            #print(f"Skipping disconnected passenger edge {edge.getID()}")
            continue

        # ActivityGen creates passenger trips. Do not include service roads
        # restricted to pedestrians, bicycles, or delivery vehicles.
        if not edge.allows("passenger"):
            #print(f"Skipping non-passenger edge {edge.getID()}")
            continue

        # A clipped network can leave one-way dead-end/fringe edges. They are
        # valid network edges, but cannot reliably serve as both trip origins
        # and destinations for ActivityGen.
        if edge.is_fringe():
            #print(f"Skipping fringe edge {edge.getID()}")
            continue

        shape = edge.getShape()
        if len(shape) < 2:
            continue

        edge_geom = LineString(shape)
        edge_geometries.append((edge.getID(), edge_geom))
        real_edges.append(edge.getID())
    print(f"Found {len(real_edges)} real edges in the network")
    print(f"Loaded {len(edge_geometries)} usable edges")

    street_populations = []
    source_population = 0.0

    for taz in block_groups_df.itertuples():

        # print(
        #     f"Processing TAZ {taz.GEOID} "
        #     f"with population {taz.population}"
        #     f" And first geometry point {taz.geometry.centroid.x}, {taz.geometry.centroid.y}"
        # )

        taz_population = taz.population
        taz_shape = taz.geometry

        edges_within_taz = []

        for edge_id, edge_geom in edge_geometries:

            # Option 1:
            # Any edge touching the TAZ counts
            if edge_geom.intersects(taz_shape):
                edges_within_taz.append(edge_id)
                #print(f"Edge {edge_id} intersects with TAZ {taz.GEOID}")

            #Option 2 (stricter):
            # if taz_shape.contains(edge_geom.centroid):
            #     edges_within_taz.append(edge_id)

        if not edges_within_taz:
            continue

        #print(f"Found {len(edges_within_taz)} edges within TAZ {taz.GEOID}")

        pop_per_edge = float(taz_population) / len(edges_within_taz)
        source_population += float(taz_population)

        for edge_id in edges_within_taz:
            street_populations.append((edge_id, pop_per_edge))

    if source_population <= 0 or not street_populations:
        raise ValueError("No census population could be assigned to network streets")

    population_scale = inhabitants / source_population
    print(
        f"Scaling mapped census population {source_population:.0f} "
        f"to requested population {inhabitants} (factor {population_scale:.4f})"
    )

    for edge_id, pop_per_edge in street_populations:
        street = doc.createElement("street")
        street.setAttribute(
            "edge",
            str(edge_id),
        )
        street.setAttribute(
            "population",
            str(pop_per_edge * population_scale),
        )
        street.setAttribute("workPosition", "1")
        streets_node.appendChild(street)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        doc.writexml(f,indent="  ",addindent="  ",newl="\n")
    with open(output_path.parent / "real_edges.txt", "w") as f:
        for edge_id in real_edges:
            f.write(f"{edge_id}\n")

    print("Finished writing:", output_path)

def assign_charging_stations_to_streets(charging_station_data_path = "..\\data\\Palo Alto\\ElectricVehicleChargingStationUsageJuly2011Dec2020_2797601782859221543.csv",
network_path = "..\\data\\Palo Alto\\PA.network.net.xml"):
    charging_stations_df = pd.read_csv(charging_station_data_path)
    
    net = sumolib.net.readNet(network_path)

    charging_station_assignments = {}

    station_locations = (
        charging_stations_df[['Station Name', 'Latitude', 'Longitude']]
        .drop_duplicates('Station Name')
        .set_index('Station Name')
        .apply(tuple, axis=1)
        .to_dict()
    )

    for station_id, (lat, lon) in station_locations.items():
        x,y = net.convertLonLat2XY(lon, lat)
        edges = sorted(net.getNeighboringEdges(x, y, 100), key=lambda e: e[1])

        for edge, distance in edges:
            print(f"Evaluating edge {edge.getID()} for charging station {station_id} at distance {distance:.2f}")
            if distance > 200:  # Adjust the threshold as needed
                continue

            valid_lane = None
            for lane in edge.getLanes():
                print(f"lane alllows passenger: {lane.allows('passenger')}, lane ID: {lane.getID()}")
                if lane.allows("passenger") and lane.getLength() >= 20 and not edge.getID().startswith(":"):
                    valid_lane = lane
                    break
            
            if valid_lane is not None:
                print(f"Assigning charging station {station_id} to lane {valid_lane.getID()} on edge {edge.getID()}")
                charging_station_assignments[station_id] = valid_lane.getID()
                break


        # if edges:
        #     closest_edge, distance = min(edges, key=lambda e: e[1])
        #     if distance <= 30:  # Adjust the threshold as needed
        #         edge_id = closest_edge.getID()
        #         edge_o = net.getEdge(edge_id)
        #         lane = edge_o.getLanes()[0] #This should be the outermost lane
        #         lane_id = lane.getID()
        #         charging_station_assignments[station_id] = lane_id

    

    doc = minidom.Document()
    root = doc.documentElement
    doc.appendChild(doc.createElement("additional"))
    for index ,item in enumerate(charging_station_assignments.items()):
        station_id, lane_id = item
        station_element = doc.createElement("chargingStation")
        #station_element.setAttribute("stationID", str(station_id))
        station_element.setAttribute("lane", str(lane_id))
        station_element.setAttribute("id", str(index))
        station_element.setAttribute("startPos", "0")
        station_element.setAttribute("endPos", "20")
        station_element.setAttribute("friendlyPos", str(True).lower())
        station_element.setAttribute("power", "100")
        station_element.setAttribute("totalPower", "1000")
        

        doc.documentElement.appendChild(station_element)
        
    output_path = "./chargers_additional.xml"
    with open(output_path, "w") as f:
        doc.writexml(f, indent="  ", addindent="  ")

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

def designate_new_vtype(route_file_path = ".\\test_PA_rou.xml", new_vtype_id = "electric_vehicle", new_vtype_params = {"vClass": "passenger", "color": "1,0,0", "accel": "3.0", "decel": "6.0", "length": "5.0", "minGap": "2.5", "maxSpeed": "13.9"}):
    # Load the existing route file
    doc = minidom.parse(route_file_path)
    root = doc.documentElement

    # Create a new vType element
    for Vehicle_type in root.getElementsByTagName("vType"):
        if Vehicle_type.getAttribute("id") == new_vtype_id:
            print(f"vType '{new_vtype_id}' already exists. Skipping creation.")
            return

    vtype_element = doc.createElement("vType")
    vtype_element.setAttribute("id", new_vtype_id)
    #vtype_element.setAttribute("has.battery.device", "true")

    # Set the attributes for the new vType
    for param, value in new_vtype_params.items():
        vtype_element.setAttribute(param, value)

    #Sumo requires weird formatting of the xml value for has.battery.device so it has to be set this way. check vtype in route files to see example
    battery_param = doc.createElement("param")
    battery_param.setAttribute("key", "has.battery.device")
    battery_param.setAttribute("value", "true")

    vtype_element.appendChild(battery_param)

    # Append the new vType to the root of the XML
    insert_before = None
    for child in root.childNodes:
        if child.nodeType != child.ELEMENT_NODE:
            continue

        if child.tagName in ("vehicle", "trip", "flow", "route"):
            insert_before = child
            break

    if insert_before is not None:
        root.insertBefore(vtype_element, insert_before)
    else:
        root.appendChild(vtype_element)

    # Save the modified route file
    with open(route_file_path, "w") as f:
        doc.writexml(f, indent="  ", addindent="  ", newl="\n")
#assign_pop_to_street()
def assign_vehicles_new_vtype(route_file_path = ".\\test_PA_rou.xml", new_vtype_id = "electric_vehicle", percent_vehicles = .15, seed = 42, new_vtype_params = {"vClass": "passenger", "color": "1,0,0", "accel": "3.0", "decel": "6.0", "length": "5.0", "minGap": "2.5", "maxSpeed": "13.9"}):
    designate_new_vtype(route_file_path, new_vtype_id,new_vtype_params)
    # Load the existing route file
    doc = minidom.parse(route_file_path)
    root = doc.documentElement

    # Get all vehicle elements
    vehicles = root.getElementsByTagName("vehicle")

    # Randomly select vehicles to change their vType
    num_vehicles = int(len(vehicles) * percent_vehicles)
    random.seed(seed)
    selected_indices = random.sample(range(len(vehicles)), min(num_vehicles, len(vehicles)))

    for index in selected_indices:
        vehicle = vehicles[index]
        vehicle.setAttribute("type", new_vtype_id)

    # Save the modified route file
    with open(route_file_path, "w") as f:
        doc.writexml(f, indent="  ", addindent="  ")
#junctions = get_junctions()
#TAZ = grab_taz(sqlite3.connect("../data/UDS.db"))




