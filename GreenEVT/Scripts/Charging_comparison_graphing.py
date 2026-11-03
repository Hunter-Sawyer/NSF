import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt


def graph_real_v_sim(real_Dict_,sim_Dict):
    hours = range(24)
    
    real = [real_Dict_.get(h, 0) for h in hours]
    sim = [sim_Dict.get(h, 0) for h in hours]

    plt.figure(figsize=(10,5))

    plt.plot(hours, real, marker='o', label='Real Data')
    plt.plot(hours, sim, marker='s', label='SUMO')

    plt.xticks(hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Energy Delivered (kWh)")
    plt.title("Hourly Charging Demand")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def load_simulation_data(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot() 
        return root
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
    
def load_real_data(file_path = "..\\data\\Palo_alto\\ElectricVehicleChargingStationUsageJuly2011Dec2020_2797601782859221543.csv"):
    df = pd.read_csv(file_path)

    df["Transaction Date (Pacific Time)"] = pd.to_datetime(
        df["Transaction Date (Pacific Time)"]
    )
    

    df["Start"] = pd.to_datetime(df["Start Date"])
    df["charging_duration"] = pd.to_timedelta(df["Charging Time (hh:mm:ss)"])
    df["End"] = df["Start"] + df["charging_duration"]

    hourly_rows = []


    for _,row in df.iterrows():

        start = row["Start"]
        end = row["End"]
        energy = row["Energy (kWh)"]

        duration_hours = row["charging_duration"].total_seconds() / 3600
        if duration_hours < 0:
            rate = energy
            hourly_rows.append({
                "Date": start.date(),
                "Hour": start.hour,
                "Energy (kWh)": rate * start.total_seconds()
            })
        else:
            rate = energy / duration_hours
        while start < end:
            next_hour = start.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)

            segment_end = min(next_hour, end)

            segment_hours = (segment_end - start).total_seconds() / 3600

            hourly_rows.append({
                "Date": start.date(),
                "Hour": start.hour,
                "Energy (kWh)": rate * segment_hours
            })

            start = segment_end

    hourly_df = pd.DataFrame(hourly_rows)

    hourly = (
        hourly_df.groupby(["Date", "Hour"])["Energy (kWh)"]
        .sum()
        .reset_index()
    )

    average_profile = (
        hourly.groupby("Hour")["Energy (kWh)"]
        .mean()
        .reindex(range(24), fill_value=0)
    )

    average_profile_wh = {
        hour: value * 1000
        for hour, value in average_profile.items()
    }

    return average_profile_wh
    # Total energy delivered each hour of each day
    # hourly = (
    #     df.groupby(["Date", "Hour"])["Energy (kWh)"]
    #     .sum()
    #     .reset_index()
    # )

    # # Average across all days
    # average_profile = (
    #     hourly.groupby("Hour")["Energy (kWh)"]
    #     .mean()
    #     .reindex(range(24), fill_value=0)
    # )

    # average_profile_wh = {
    #     hour: value * 1000
    #     for hour, value in average_profile.items()
    # }
    

    # return average_profile_wh



def extract_demand_by_hour(root):
    hourly_energy = defaultdict(float)

    for station in root.findall("chargingStation"):
        for vehicle in station.findall("vehicle"):
            for step in vehicle.findall("step"):

                time = float(step.get("time"))
                energy = float(step.get("energyCharged"))

                hour = int(time // 3600)

                hourly_energy[hour] += energy

    return dict(hourly_energy)

def main():
    # Load the data from the XML file
    s_data = load_simulation_data("charging_stations_output.xml")
    s_data_cleaned = extract_demand_by_hour(s_data)
    r_data_cleaned = load_real_data()




    

    # Perform some analysis or graphing here
    graph_real_v_sim(r_data_cleaned,s_data_cleaned)
    #print(r_data_cleaned)


if __name__ == "__main__":
    main()