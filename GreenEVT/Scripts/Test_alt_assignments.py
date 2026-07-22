import activity_gen_assign_populations
import geopandas as gpd
import pandas as pd

if __name__ == "__main__":
    # Example usage
    database_path = "../data/UDS.db"
    shapefile_path = "..\\..\\Cali_traffic_uknown\\Annual_average_daily_traffic.shp"
    network_path = "..\\data\\Palo Alto\\PA.network.net.xml"
    other_test_file = "..\\..\\tl_2019_06_bg\\tl_2019_06_bg.shp"
    geoID_path = "..\\..\\tl_2019_06_bg\\block_groups_pop\\ACSDT5Y2020.B01003-Data.csv"

    test_geo = gpd.read_file(other_test_file)
    geoIDs = pd.read_csv(geoID_path)


    # test_geo = test_geo['GEOID'].tolist()

    # for geoID in test_geo:
    #     if test_value in geoID:
    #         print(f"Found {test_value} in GEOID: {geoID}")
    # #print(test_geo['GEOID'])
    #print(geoIDs.head())
    # geo_id_list = geoIDs['GEO_ID'].tolist()
    # sublist = [s[-12:] for s in geo_id_list]
    #print(geoIDs.head())


    
    geo_id_list = geoIDs['GEO_ID'].tolist()
    sublist = [s[-12:] for s in geo_id_list]

    # Create matching GEOID column in the CSV
    geoIDs['GEOID'] = geoIDs['GEO_ID'].str[-12:]

    # Create lookup dictionary
    pop_lookup = dict(
        zip(
            geoIDs['GEOID'],
            geoIDs['B01003_001E']
        )
    )

    #print(geoIDs['B01003_001E'].head())

    # Add population column to test_geo
    test_geo['population'] = test_geo['GEOID'].map(pop_lookup)
    test_geo = test_geo.dropna(subset=['population'])  # Drop rows with NaN population values
    print(test_geo.head())

    #conn = sqlite3.connect(database_path)
    





    
