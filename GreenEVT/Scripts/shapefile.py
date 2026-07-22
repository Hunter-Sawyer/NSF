import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

import string
import os,sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import sumolib
import logging
import numpy as np
import sqlite3
import math
from xml.dom import minidom
import os
from tqdm import tqdm
import random

def generate_radial_triangles(center, radius, n_triangles):
    cx, cy = center
    triangles = []

    angle_step = 2 * math.pi / n_triangles

    for i in range(n_triangles):
        theta1 = i * angle_step
        theta2 = (i + 1) * angle_step

        p1 = (cx + radius * math.cos(theta1),
              cy + radius * math.sin(theta1))

        p2 = (cx + radius * math.cos(theta2),
              cy + radius * math.sin(theta2))
        print(f"Triangle {i+1}: Center: {center}, Point 1: {p1}, Point 2: {p2}")
        triangles.append((center, p1, p2))

    return triangles
def point_in_triangle(px, py, v1, v2, v3):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign((px, py), v1, v2) < 0.0
    b2 = sign((px, py), v2, v3) < 0.0
    b3 = sign((px, py), v3, v1) < 0.0

    return (b1 == b2) and (b2 == b3)

def include_triangles(triangles,shapefile_path = "..\\..\\NCDOT_2024_Traffic_Segment_Shapefile_Description\\NCDOT_2024_AADT_Segments.shp", column=None, cmap='viridis', legend=True,
    network_path = "../data/sumo_network/greensboro.net.xml"):

    net = sumolib.net.readNet(network_path)
    doc = minidom.Document()
    root = doc.documentElement

    gdf = gpd.read_file(shapefile_path)

    gdf_LL = gdf.to_crs(epsg=4326)
    print(gdf_LL.crs)

    print(gdf_LL.head()[gdf.geometry.name])
    bbox = gpd.GeoDataFrame(geometry=[box(-80.62,35.676,-79.2,37)], crs=gdf.crs)

    gdf_LL = gpd.clip(gdf_LL, bbox)
    road_edge = []
    road_edge1 = []
    road_edge2 = []
    road_edge3 = []
    road_edge4 = []
    road_edge5 = []

    for road in gdf_LL.itertuples():
        x,y = net.convertLonLat2XY(road.geometry.centroid.x, road.geometry.centroid.y)
        edges = net.getNeighboringEdges(x,y, 10)
        if len(edges) > 0:
            #print(f"Found {len(edges)} neighboring edges in SUMO network:")
            sorted_edges = sorted([(dist,edge) for edge,dist in edges], key=lambda x:x[0])
            #print(f"Closest edge is {sorted_edges[0][1]} at distance {sorted_edges[0][0]} meters")
            if point_in_triangle(x,y,triangles[0][0],triangles[0][1],triangles[0][2]):
                print(f"Shapefile road segment {road} is within the first triangle and will be excluded.")
                road_edge1.append((road, sorted_edges[0][1].getID()))
            
            if point_in_triangle(x,y,triangles[1][0],triangles[1][1],triangles[1][2]):
                print(f"Shapefile road segment {road} is within the second triangle and will be excluded.")
                road_edge2.append((road, sorted_edges[0][1].getID()))
            
            if point_in_triangle(x,y,triangles[2][0],triangles[2][1],triangles[2][2]):
                print(f"Shapefile road segment {road} is within the third triangle and will be excluded.")
                road_edge3.append((road, sorted_edges[0][1].getID()))
            
            if point_in_triangle(x,y,triangles[3][0],triangles[3][1],triangles[3][2]):
                print(f"Shapefile road segment {road} is within the fourth triangle and will be excluded.")
                road_edge4.append((road, sorted_edges[0][1].getID()))

            if point_in_triangle(x,y,triangles[4][0],triangles[4][1],triangles[4][2]):
                print(f"Shapefile road segment {road} is within the fifth triangle and will be excluded.")
                road_edge5.append((road, sorted_edges[0][1].getID()))
            

            
        else:
            print("No neighboring edges found in SUMO network.")

    with open("matched_edges_first_triangle_inclusive.txt","w") as f:
        for road, edge_id in road_edge1:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_second_triangle_inclusive.txt","w") as f:
        for road, edge_id in road_edge2:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_third_triangle_inclusive.txt","w") as f:
        for road, edge_id in road_edge3:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_fourth_triangle_inclusive.txt","w") as f:
        for road, edge_id in road_edge4:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_fifth_triangle_inclusive.txt","w") as f:
        for road, edge_id in road_edge5:
            f.write(f"{road.AADT}:{edge_id}\n")
    
    #with open("matched_edges.txt","w") as f:
    #    for road, edge_id in road_edge:
    #        f.write(f"{road.AADT}:{edge_id}\n")


def exclude_triangles(triangles,shapefile_path = "..\\..\\NCDOT_2024_Traffic_Segment_Shapefile_Description\\NCDOT_2024_AADT_Segments.shp", column=None, cmap='viridis', legend=True,
    network_path = "../data/sumo_network/greensboro.net.xml"):

    net = sumolib.net.readNet(network_path)
    doc = minidom.Document()
    root = doc.documentElement

    gdf = gpd.read_file(shapefile_path)

    gdf_LL = gdf.to_crs(epsg=4326)
    print(gdf_LL.crs)

    print(gdf_LL.head()[gdf.geometry.name])
    bbox = gpd.GeoDataFrame(geometry=[box(-80.62,35.676,-79.2,37)], crs=gdf.crs)

    gdf_LL = gpd.clip(gdf_LL, bbox)
    road_edge = []
    road_edge1 = []
    road_edge2 = []
    road_edge3 = []
    road_edge4 = []
    road_edge5 = []

    for road in gdf_LL.itertuples():
        x,y = net.convertLonLat2XY(road.geometry.centroid.x, road.geometry.centroid.y)
        edges = net.getNeighboringEdges(x,y, 10)
        if len(edges) > 0:
            #print(f"Found {len(edges)} neighboring edges in SUMO network:")
            sorted_edges = sorted([(dist,edge) for edge,dist in edges], key=lambda x:x[0])
            #print(f"Closest edge is {sorted_edges[0][1]} at distance {sorted_edges[0][0]} meters")
            if not point_in_triangle(x,y,triangles[0][0],triangles[0][1],triangles[0][2]):
                print(f"Shapefile road segment {road} is within the first triangle and will be excluded.")
                road_edge1.append((road, sorted_edges[0][1].getID()))
            
            if not point_in_triangle(x,y,triangles[1][0],triangles[1][1],triangles[1][2]):
                print(f"Shapefile road segment {road} is within the second triangle and will be excluded.")
                road_edge2.append((road, sorted_edges[0][1].getID()))
            
            if not point_in_triangle(x,y,triangles[2][0],triangles[2][1],triangles[2][2]):
                print(f"Shapefile road segment {road} is within the third triangle and will be excluded.")
                road_edge3.append((road, sorted_edges[0][1].getID()))
            
            if not point_in_triangle(x,y,triangles[3][0],triangles[3][1],triangles[3][2]):
                print(f"Shapefile road segment {road} is within the fourth triangle and will be excluded.")
                road_edge4.append((road, sorted_edges[0][1].getID()))

            if not point_in_triangle(x,y,triangles[4][0],triangles[4][1],triangles[4][2]):
                print(f"Shapefile road segment {road} is within the fifth triangle and will be excluded.")
                road_edge5.append((road, sorted_edges[0][1].getID()))
            

            
        else:
            print("No neighboring edges found in SUMO network.")

    with open("matched_edges_first_triangle.txt","w") as f:
        for road, edge_id in road_edge1:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_second_triangle.txt","w") as f:
        for road, edge_id in road_edge2:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_third_triangle.txt","w") as f:
        for road, edge_id in road_edge3:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_fourth_triangle.txt","w") as f:
        for road, edge_id in road_edge4:
            f.write(f"{road.AADT}:{edge_id}\n")
    with open("matched_edges_fifth_triangle.txt","w") as f:
        for road, edge_id in road_edge5:
            f.write(f"{road.AADT}:{edge_id}\n")
    
    #with open("matched_edges.txt","w") as f:
    #    for road, edge_id in road_edge:
    #        f.write(f"{road.AADT}:{edge_id}\n")


def plot_shapefile(shapefile_path = "..\\..\\NCDOT_2024_Traffic_Segment_Shapefile_Description\\NCDOT_2024_AADT_Segments.shp", column=None, cmap='viridis', legend=True,
    network_path = "../data/sumo_network/greensboro.net.xml"):
    net = sumolib.net.readNet(network_path)

    doc = minidom.Document()
    #NetDom = minidom.parse(network_path)
    root = doc.documentElement

    #edges = list(NetDom.getElementsByTagName("edge"))

    gdf = gpd.read_file(shapefile_path)
    

    #print(gdf.head())
    #print(gdf.crs)
    gdf_LL = gdf.to_crs(epsg=4326)
    print(gdf_LL.crs)

    print(gdf_LL.head()[gdf.geometry.name])
    #print(gdf.iloc[0].geometry)
    #print(f"rows in the GeoDataFrame: {len(gdf)}")

    #gdf_sample = gdf.sample(n=1000, random_state=1)
    bbox = gpd.GeoDataFrame(geometry=[box(-80.62,35.676,-79.2,37)], crs=gdf.crs)



    gdf_LL = gpd.clip(gdf_LL, bbox)
    road_edge = []
    for road in gdf_LL.itertuples():
        #print(road)
        #print(f"road.geometry.centroid.x, road.geometry.centroid.y: {road.geometry.centroid.x, road.geometry.centroid.y}")
        x,y = net.convertLonLat2XY(road.geometry.centroid.x, road.geometry.centroid.y)
        #print(f"Converted to SUMO coordinates: {x,y}")
        edges = net.getNeighboringEdges(x,y, 10)
        if len(edges) > 0:
            print(f"Found {len(edges)} neighboring edges in SUMO network:")
            sorted_edges = sorted([(dist,edge) for edge,dist in edges], key=lambda x:x[0])
            print(f"Closest edge is {sorted_edges[0][1]} at distance {sorted_edges[0][0]} meters")
            road_edge.append((road, sorted_edges[0][1].getID()))
        else:
            print("No neighboring edges found in SUMO network.")

    #for road, edge_id in road_edge:
        #print(f"Shapefile road segment {road} matched to SUMO edge ID: {edge_id}")
        #print(road.AADT)
    #print(len(road_edge))
    #dup = [t[0] for t in road_edge]
    #print(f"Number of unique shapefile road segments matched: {len(set(dup))}")
    #string = ""
    #for road, edge_id in road_edge:
    #    string += f"{edge_id} "
    #print("Matched SUMO edge IDs:")
    #print(string)
        
    with open("matched_edges.txt","w") as f:
        for road, edge_id in road_edge:
            f.write(f"{road.AADT}:{edge_id}\n")

            

    #gdf_LL.plot(figsize=(10,8),edgecolor='black',alpha=.6)
    #plt.title('Shapefile Plot')
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    #plt.show()

def exclusion_percents(exclusion_percent=0.2, shapefile_path = "..\\..\\NCDOT_2024_Traffic_Segment_Shapefile_Description\\NCDOT_2024_AADT_Segments.shp", column=None, cmap='viridis', legend=True,
network_path = "../data/sumo_network/greensboro.net.xml",seed=42,bbox_coords=box(-80.62,35.676,-79.2,37)):
    net = sumolib.net.readNet(network_path)
    random.seed(seed)

    doc = minidom.Document()
    root = doc.documentElement

    xmin, ymin, xmax, ymax = net.getBoundary()

    lon1, lat1 = net.convertXY2LonLat(xmin, ymin)
    lon2, lat2 = net.convertXY2LonLat(xmax, ymax)

    print(lon1, lat1, lon2, lat2)

    bbox = box(
        min(lon1, lon2),
        min(lat1, lat2),
        max(lon1, lon2),
        max(lat1, lat2)
    )


    gdf = gpd.read_file(shapefile_path)
    gdf_LL = gdf.to_crs(epsg=4326)
    print(gdf_LL.crs)

    print(gdf_LL.head()[gdf.geometry.name])
    bbox = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")



    gdf_LL = gpd.clip(gdf_LL, bbox)
    road_edge = []
    for road in gdf_LL.itertuples():
        x,y = net.convertLonLat2XY(road.geometry.centroid.x, road.geometry.centroid.y)
        edges = net.getNeighboringEdges(x,y, 10)
        if len(edges) > 0:
            print(f"Found {len(edges)} neighboring edges in SUMO network:")
            sorted_edges = sorted([(dist,edge) for edge,dist in edges], key=lambda x:x[0])
            print(f"Closest edge is {sorted_edges[0][1]} at distance {sorted_edges[0][0]} meters")
            if random.random() >= exclusion_percent:  # Exclude the specified percentage of the matched edges
                road_edge.append((road, sorted_edges[0][1].getID()))
        else:
            print("No neighboring edges found in SUMO network.")
        
    with open(f"matched_edges_{int(exclusion_percent*100)}_excluded.txt","w") as f:
        for road, edge_id in road_edge:
            #print(f"Shapefile road segment {road} matched to SUMO edge ID: {edge_id}")
            print(road.AHEAD_AADT)
            if road.AHEAD_AADT is not None:
                f.write(f"{int(road.AHEAD_AADT)}:{edge_id}\n")
            #f.write(f"{road.AADT}:{edge_id}\n")


#same as exclusion_percents buth for a geojson instead of a shapefile
def exclusion_percents_GeoJson(exclusion_percent=0.0, geojson_path = "..\\data\\Palo Alto\\Traffic_volumes_AADT.geojson", 
column=None, cmap='viridis', legend=True,network_path = "..\\data\\Palo Alto\\PA.network.net.xml",seed=42):
    net = sumolib.net.readNet(network_path)
    random.seed(seed)

    doc = minidom.Document()
    root = doc.documentElement


exclusion_percents(0.0,shapefile_path = "..\\..\\Cali_traffic_uknown\\Annual_average_daily_traffic.shp", network_path = "..\\data\\Palo Alto\\PA.network.net.xml",bbox_coords=box(-122.25,37.35,-122.0,37.5))
#exclusion_percents(0.35,seed=891)
#generate_radial_triangles(center=(40000,35888), radius=65000-35888, n_triangles=5)
#plot_shapefile()
#center_point = [40000,35888]
#distance_From_center = 65000-35888
#angles = 72
#triangles = generate_radial_triangles(center=(40000,35888), radius=65000-35888, n_triangles=5)
#exclusion_percents(.55)
#exclusion_percents(.75)
#include_triangles(triangles)
#exclude_triangles(triangles)
