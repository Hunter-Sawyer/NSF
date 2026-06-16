import os
import sys
import random
import subprocess
from xml.dom import minidom
import activity_gen_assign_populations
import correlation_test
import numpy as np
import math
from pathlib import Path
import shutil
import gzip

from concurrent.futures import ProcessPoolExecutor
from functools import partial

def gzip_file_safely(file_path, remove_original=True, overwrite=True):
    gzip_path = file_path + ".gz"
    temp_gzip_path = gzip_path + ".tmp"

    if not os.path.exists(file_path):
        print(f"[WARNING] Cannot gzip missing file: {file_path}")
        return False

    if os.path.exists(gzip_path) and not overwrite:
        print(f"[INFO] Gzip file already exists, skipping: {gzip_path}")
        return True

    try:
        # Write the new gzip to a temporary file first
        with open(file_path, "rb") as source_file:
            with gzip.open(temp_gzip_path, "wb") as gzip_file:
                shutil.copyfileobj(source_file, gzip_file)

        # Replace old .gz only after the new .gz.tmp was written successfully
        os.replace(temp_gzip_path, gzip_path)

        if remove_original:
            os.remove(file_path)

        return True

    except Exception as e:
        print(f"[WARNING] Failed to gzip {file_path}: {type(e).__name__}: {e}")

        if os.path.exists(temp_gzip_path):
            os.remove(temp_gzip_path)

        return False
    

def directory_creation(name):
    if not os.path.exists(f"../genetic_alg/{name}"):
        os.mkdir(f"../genetic_alg/{name}")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/logs")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/outputs")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/routes")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/stats")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/trackers")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/trips_routes")
        os.mkdir(f"../genetic_alg/{name}/static_files")
        os.mkdir(f"../genetic_alg/{name}/intermediate_files/arrays")
        os.mkdir(f"../genetic_alg/{name}/save_dir")
        
    else:
        print("Directory exists")

def tracker_assignment(child_index, generation_index,SDN2 = ""):
    tracker_name = f"../outputs/{child_index}_{generation_index}_traffic.xml"
    
    doc = minidom.parse("../genetic_alg/static_files/Route_finder.xml")
    
    edge = doc.getElementsByTagName("edgeData")[0]
    edge.setAttribute("file",tracker_name)
    with open(f"../genetic_alg/{SDN2}intermediate_files/trackers/{child_index}_{generation_index}_tracker.xml","w") as f:
        doc.writexml(f)
    return


# default seed 99
#90 - 
def process_child(child_index,generation_index,IO_array, job_array, real_edges,gates,SDN2 = "",road_activity = r"./matched_edges.txt"):
    #print(f"Processing Child {child_index}...")
    activity_gen_assign_populations.assign_jobs(output_file_name=f"../genetic_alg/{SDN2}intermediate_files/stats/Child_{child_index}_{generation_index}.xml",job_assignments=job_array[child_index,:],real_edges=real_edges,n_gates=gates[child_index, :, :],IO_list = IO_array[child_index])
    #print(f"Assigned Jobs")
    
    T_R_Command = f"activitygen --net-file ../data/sumo_network/greensboro.net.xml --stat-file ../genetic_alg/{SDN2}intermediate_files/stats/Child_{child_index}_{generation_index}.xml --output-file ../genetic_alg/{SDN2}intermediate_files/trips_routes/{child_index}_{generation_index}_trips.rou.xml --seed 93 > NUL 2>&1"
    #print(f"Running command: {T_R_Command}")
    
    os.system(f"{T_R_Command}")

    T_R_Z_Command = f"gzip ../genetic_alg/{SDN2}intermediate_files/stats/Child_{child_index}_{generation_index}.xml"
    
    TR_path = Path(f"../genetic_alg/{SDN2}intermediate_files/stats/Child_{child_index}_{generation_index}.xml")
    
    TR_Z_path = Path(f"../genetic_alg/{SDN2}intermediate_files/stats/Child_{child_index}_{generation_index}.xml.gz")

# Check if the file exists
    
        
        
    #print(f"ran activity_gen")
    #print(f"Generated trips file for Child {child_index}_{generation_index}.xml")
    R_Command = f"duarouter --net-file ../data/sumo_network/greensboro.net.xml --route-files ../genetic_alg/{SDN2}intermediate_files/trips_routes/{child_index}_{generation_index}_trips.rou.xml --output-file ../genetic_alg/{SDN2}intermediate_files/routes/{child_index}_{generation_index}_rou.xml --ignore-errors > NUL 2>&1"
    os.system(f"{R_Command}")
    
    #print(f"Generated routes for Child {child_index}_{generation_index}.xml")

    tracker_assignment(child_index,generation_index,SDN2 = SDN2)

    S_Command = f"sumo --net-file ../data/sumo_network/greensboro.net.xml --route-files ../genetic_alg/{SDN2}intermediate_files/routes/{child_index}_{generation_index}_rou.xml --additional-files ../genetic_alg/{SDN2}intermediate_files/trackers/{child_index}_{generation_index}_tracker.xml > NUL 2>&1"
    os.system(f"{S_Command}")
    
    if TR_path.exists() and not TR_Z_path.exists():
    
        os.system(f"{T_R_Z_Command}")
    
    
    T_R_R_Command = (f"rm ../genetic_alg/{SDN2}intermediate_files/stats/Child_{child_index}_{generation_index}.xml")
    if TR_path.exists():
        os.system(f"{T_R_R_Command}")
    
    #> NUL 2>&1
    #print(f"SUMO simulation completed for Child {child_index}_{generation_index}.xml")
    output_file = f"../genetic_alg/{SDN2}intermediate_files/outputs/{child_index}_{generation_index}_traffic.xml"
    none,none,correlations = correlation_test.test_correlation(road_activity=road_activity,output_path=output_file)
    
    gzip_file_safely(output_file, remove_original=True, overwrite=True)
    #print(f"Correlation for Child {child_index}_{generation_index}.xml: {correlations}")
    #print(f"Child {child_index} correlation: {correlations}")
    return correlations

def crossover(job_array,fitness_scores):
    max_fitness_index = np.argmax(fitness_scores)
    parent1 = job_array[max_fitness_index,:]
    crossover_point = np.random.randint(0,2,size = (len(parent1)))
    #print(sum(crossover_point))
    #print(f"Crossover point: {crossover_point[:5]}")
    for i in range(job_array.shape[0]):
        if i != max_fitness_index:
            #print(f"Child {i} before crossover: {job_array[i,:5]}")
            #print(f"Parent1: {parent1[:5]}")
            
            parent2 = job_array[i,:]
            #print(f"Parent2: {parent2[:5]}")
            mask = np.multiply(parent1,crossover_point)
            denom = crossover_point+1
            mask = np.add(mask,parent2)
            mask = np.divide(mask,denom)
            job_array[i,:] = mask

            #print(f"Child {i} after crossover: {job_array[i,:5]}")
            
            #print(f"denominator: {denom[:5]}")
            #print(f"mask: {mask[:5]}")
            
def crossover_ratio(job_array, fitness_scores, gate_array, ratio=0.35, crossover_Modifier=1.0):
    ratio = max(0.0, min(1.0, ratio * crossover_Modifier))

    # Only choose from valid fitness scores
    if np.all(np.isnan(fitness_scores)):
        raise ValueError("All fitness scores are NaN; cannot choose crossover parent.")

    max_fitness_index = np.nanargmax(fitness_scores)

    parent1_jobs = job_array[max_fitness_index, :].copy()
    parent1_gates = gate_array[max_fitness_index].reshape(-1).copy()

    ratio_off = 1.0 - ratio

    for i in range(job_array.shape[0]):
        if i == max_fitness_index:
            continue

        # New job mask for this child
        crossover_point = np.random.choice(
            [0, 1],
            size=parent1_jobs.shape[0],
            p=[ratio_off, ratio]
        )

        parent2_jobs = job_array[i, :].copy()

        # If crossover_point[j] == 1, average parent1 and parent2.
        # If crossover_point[j] == 0, keep parent2.
        job_array[i, :] = (
            parent1_jobs * crossover_point + parent2_jobs
        ) / (crossover_point + 1)

        # New gate mask for this child
        parent2_gates = gate_array[i].reshape(-1).copy()

        crossover_gates = np.random.choice(
            [0, 1],
            size=parent1_gates.shape[0],
            p=[ratio_off, ratio]
        )

        child_gates = (
            parent1_gates * crossover_gates + parent2_gates
        ) / (crossover_gates + 1)

        gate_array[i] = child_gates.reshape(gate_array[i].shape)
        

def IO_crossover(IO_array, fitness_scores, crossover_Modifier=1.0, angle=0.1):

    MFI = np.nanargmax(fitness_scores)
    parent_one = IO_array[MFI]

    parent_one_I = parent_one[0]
    parent_one_O = parent_one[1]

    for i in range(IO_array.shape[0]):

        # keep elite unchanged
        if i == MFI:
            continue

        parent_two = IO_array[i]
        parent_two_I = parent_two[0]
        parent_two_O = parent_two[1]

        # crossover weights
        c1 = random.uniform(1-angle, 1+angle) * crossover_Modifier
        c3 = random.uniform(1-angle, 1+angle) * crossover_Modifier

        c2 = 2 - c1
        c4 = 2 - c3

        child_I = (parent_one_I * c1 + parent_two_I * c2) / 2
        child_O = (parent_one_O * c3 + parent_two_O * c4) / 2

        IO_array[i,0] = max(0,int(child_I))
        IO_array[i,1] = max(0,int(child_O))

    return IO_array
    
def IO_mutation(IO_array, fitness_scores, mutation_rate = 0.1,mutation_magnitude_mod = 1.0):
    
    MFI = np.nanargmax(fitness_scores)
    parent_one = IO_array[MFI]
    
    for i in range(IO_array.shape[0]):

        # keep elite unchanged
        if i == MFI:
            continue
        if random.random() < mutation_rate:
            IO_array[i][0] = max(0,IO_array[i][0] *random.uniform(.5, 1.5) * mutation_magnitude_mod)
            IO_array[i][1] = max(0,IO_array[i][1] *random.uniform(.5, 1.5) * mutation_magnitude_mod)
        
    return

def mutation(job_array,fitness_scores,gate_array,mutation_rate=0.1):
    num_children, num_genes = job_array.shape
    max_fitness_index = np.nanargmax(fitness_scores)
    num_gates = gate_array.shape[1]

    for i in range(num_children):
        if i != max_fitness_index:
            for j in range(num_genes):
                if random.random() < mutation_rate:
                    old_value = job_array[i,j]
                    job_array[i,j] = max(0,job_array[i,j] * random.uniform(0,2) + random.uniform(-3,3)) 
                    #print(f"Mutated Child {i}, Gene {j} from {old_value} to {job_array[i,j]}")
            for z in range(num_gates):
                for k in range(2):
                    if random.random() < mutation_rate:

                        old_value = gate_array[i, z, k]
                        gate_array[i, z, k] = max(0,gate_array[i, z, k] * random.uniform(0, 2) + random.uniform(-3, 3))
                        
def save_state(dir_name,generation,total_generations,i_M_M,mutation_rate,crossover,n_CC,n_gates,job_array,fitness_Scores,job_MR = 1.0,io_MR= 1.0,gate_MR = 1.0):
    non_list_data = f"../genetic_alg/{dir_name}/save_dir/save_file.txt"
    n_gates_file_name = f"../genetic_alg/{dir_name}/save_dir/n_gates.npy"
    job_array_file_name = f"../genetic_alg/{dir_name}/save_dir/job_array.npy"
    #historic_fitness_file_name = f"../genetic_alg/{dir_name}/save_dir/historic_fitness.txt"
    fitness_scores_file_name = f"../genetic_alg/{dir_name}/save_dir/fitness_scores.npy"
    
    with open(non_list_data, 'w') as file:
            file.write("%s\n" %generation)
            file.write("%s\n" %total_generations)
            file.write("%s\n" %i_M_M)
            file.write("%s\n" %mutation_rate)
            file.write("%s\n" %crossover)
            file.write("%s\n" %n_CC)
        
    #with open(fitness_scores_file_name,'w') as file:
    #    for item in historic_fitness:
    #        file.write("%s\n"%item)
    
    
    np.save(str(n_gates_file_name),n_gates)
    np.save(str(job_array_file_name),job_array)
    np.save(str(fitness_scores_file_name),fitness_Scores)
    
    return
        
def load_state(dir_name):
    non_list_data = f"../genetic_alg/{dir_name}/save_dir/save_file.txt"
    n_gates_file_name = f"../genetic_alg/{dir_name}/save_dir/n_gates.npy"
    job_array_file_name = f"../genetic_alg/{dir_name}/save_dir/job_array.npy"
    historic_fitness_file_name = f"../genetic_alg/{dir_name}/save_dir/historic_fitness.txt"
    fitness_scores_file_name = f"../genetic_alg/{dir_name}/save_dir/fitness_scores.npy"
    
    with open(non_list_data, 'r') as file:
        generation = int(file.readline().strip())
        total_generations = int(file.readline().strip())
        i_M_M = float(file.readline().strip())
        mutation_rate = float(file.readline().strip())
        crossover = float(file.readline().strip())
        n_CC = int(file.readline().strip())
        
    #with open(historic_fitness_file_name,'r') as file:
    #    historic_fitness = [float(item.strip()) for item in file]
    
    n_gates = np.load(str(n_gates_file_name))
    job_array = np.load(str(job_array_file_name))
    fitness_Scores = np.load(str(fitness_scores_file_name))
    #historic_fitness = []
    
    return generation,total_generations,i_M_M,mutation_rate,crossover,n_CC,n_gates,job_array,fitness_Scores

def main():

    Re_calc_pop_var = False
    arguments = sys.argv[1:]
    if "--recalc" in arguments:
        Re_calc_pop_var = true
    
    if "--dir" in arguments:
        SDN = arguments[arguments.index("--dir")+1]
        print(f"sub directory = {SDN}")
        SDN2 = SDN + "/"
        directory_creation(SDN)
    else:
        SDN2 = ""
        
    if "--pop" in arguments:
        pop = int(arguments[arguments.index("--pop")+1])
        Re_calc_pop_var = True
        print(f"pop = {pop}")
    else:
        pop = 1000
        Re_calc_pop_var = False
        print(f"pop = {pop}")
        
    if "--crossover" in arguments:
        crossover_rate = float(arguments[arguments.index("--crossover")+1])
        #Re_calc_pop_var = True
        print(f"crossover = {crossover_rate}")
    else:
        crossover_rate = .3
        print(f"crossover_rate = {crossover_rate}")
        
    if "--mutation" in arguments:
        mutation_rate = float(arguments[arguments.index("--mutation")+1])
        #Re_calc_pop_var = True
        print(f"mutation_rate = {mutation_rate}")
    else:
        mutation_rate = .1
        print(f"mutation_rate = {mutation_rate}")
        
    if "--alt_matched" in arguments:
        alt_file = arguments[arguments.index("--alt_matched")+1]
        train_set = True
    else:
        train_set = False
        

    
    if "--children" in arguments:
        num_children = int(arguments[arguments.index("--children")+1])
        print(f"Num Children = {num_children}")
    else:
        num_children = 8
        print(f"Num Children = {num_children}")
        
    if "--gen" in arguments:
        num_generations = int(arguments[arguments.index("--gen")+1])
        print(f"Num Gen = {num_generations}")
    else:
        num_generations = 35
        print(f"Num generations = {num_generations}")
        
    if "--restart" in arguments:
        restart = True
    else:
        restart = False
        current_gen = 0
        print(f"start gen = {current_gen}")

    #print(f"../genetic_alg/{SDN2}static_files/pop_file.xml")
        

    #default 42
    #35 - 37
    np.random.seed(38)
    random.seed(38)
    
    if not os.path.exists(f"../genetic_alg/{SDN2}static_files/pop_file.xml") or not os.path.exists("../genetic_alg/static_files/real_edges.txt")or Re_calc_pop_var:
        print("Generating population file...")
        activity_gen_assign_populations.assign_pop_to_street_without_jobs_init(output_path = f"../genetic_alg/{SDN2}static_files/pop_file.xml",inhabitants=pop)
    else:
        print("Population file already exists. Skipping generation.")
    #if Re_calc_pop_var:
        
        
    real_edges = []
    with open("../genetic_alg/static_files/real_edges.txt","r") as f:
        real_edges = f.read().splitlines()
        print(f"Found {len(real_edges)} real edges.")
        print(real_edges[:5])

    n_gates = np.random.randint(0,10,size=(num_children,66,2))
    IO_list = [pop*.15,pop*.15]
    IO_array = np.tile(IO_list,(num_children,1))

    print("Computing initial Job states")

    if restart:
        print("Restart Job")
        generation,total_generations,I_M_M,mutation_rate,crossover,n_CC,n_gates,job_array,fitness_Scores = load_state(SDN)
        current_gen = generation
        num_generation = total_generations
        inertial_modifier_mutation= I_M_M
        mutation_rate = mutation_rate
        crossover = crossover_rate
        no_change_count = n_CC
        #historical_Fitness= historic_fitness
        n_gates = n_gates
        job_array = job_array
        Fitness_scores = fitness_Scores
        
        
        
    else:
        job_array = np.random.randint(0,10,size=(num_children,len(real_edges)))
        historical_Fitness = []
        inertial_modifier_mutation = 0.7
        inertial_crossover_mutation = 1.0
        inertia = 1.0
        #max_fitness_scores = [0.0]
        no_change_count = 1
    
    print("Initial Job states computed.")
    print(f"job array shape: {job_array.shape}")
    

    print(job_array[0,:5])
    print("Initial Gate states computed")
    print(f"gate array shape: {n_gates.shape}")
    Fitness_scores = np.zeros(num_children,)

    #crossover(job_array,Fitness_scores)
    #print("Initial crossover completed.")
    #print(job_array[0,:5])

    

    generation = current_gen
    previous_best_index = None
    previous_best_score = None
    
    while generation < num_generations:
        mutation_r_s = math.sin(.7 * math.pi * generation)/2.4 + .5
        print(f"Starting Generation {generation}...")
        # save_state(dir_name=SDN,
        # generation=generation,
        # total_generations=num_generations,
        # i_M_M=inertial_modifier_mutation,
        # mutation_rate=mutation_rate,
        # crossover=crossover_rate,
        # n_CC=no_change_count,
        # n_gates=n_gates,
        # job_array=job_array,
        # #historic_fitness=historical_Fitness,
        # fitness_Scores=Fitness_scores,
        # job_MR = 1.0,io_MR= 1.0,gate_MR = 1.0)
        if not train_set:
            worker = partial(process_child, generation_index=generation,IO_array = IO_array, job_array=job_array, real_edges=real_edges,gates = n_gates,SDN2 = SDN2,road_activity = r"./matched_edges.txt")
        else:
            worker = partial(process_child, generation_index=generation,IO_array = IO_array, job_array=job_array, real_edges=real_edges,gates = n_gates,SDN2 = SDN2,road_activity = alt_file)
            
        max_retries = 3
        retry_count = 0
        results = None
        
        #with ProcessPoolExecutor(max_workers=33) as executor:
        #    results = list(executor.map(worker, range(num_children)))
                    
        while retry_count < max_retries:
            try:
                print(f" retry status ({retry_count}/{max_retries})")
                with ProcessPoolExecutor(max_workers=33) as executor:
                    results = list(executor.map(worker, range(num_children)))
                    break
            except FileNotFoundError as e:
                print(f"[WARNING] Missing SUMO file. Retrying generation... ({retry_count+1}/{max_retries})")
                print(e)
                retry_count += 1
            except Exception as e:
                print(f"Other error: {type(e).__name__}: {e}")
                
        if results is None:
            raise RuntimeError("Generation failed after max retries. Aborting.")
        
        Fitness_scores = np.array(results)
        
        
        if previous_best_index is not None:
            repeated_score = Fitness_scores[previous_best_index]
            
            #lower check
            if np.isnan(repeated_score) or repeated_score < previous_best_score:
                print(
                    f"[WARNING] Generation {generation} failed elite check. "
                    f"Previous best child {previous_best_index} had score "
                    f"{previous_best_score}, but now returned NaN. "
                    f"Repeating Generation {generation}."
                )
                continue
        
        max_fitness_index = np.nanargmax(Fitness_scores)
        max_Fitness = np.nanmax(Fitness_scores)
        
        print(f"Fitness scores for Generation {generation}: {Fitness_scores}")
        print(f"Best performing child in Generation {generation}: Child {max_fitness_index}")
        print(f"With fitness score: {Fitness_scores[max_fitness_index]}")
        print(f"With average fitness between children of: {np.mean(Fitness_scores)}")
        print(f"With std dev of: {np.std(Fitness_scores)}")
        crossover_ratio(job_array,Fitness_scores,n_gates,crossover_rate)
        IO_crossover(IO_array, Fitness_scores, crossover_Modifier=1.0, angle=0.1)
        print(f"Crossover completed for Generation {generation}.")
        print(f"Mutation rate of: {mutation_rate}. Mutation modifier of: {inertial_modifier_mutation}. Combined value = {max(.1,min(mutation_rate*inertial_modifier_mutation,.90))}")
        mutation(job_array,Fitness_scores,n_gates,mutation_rate=max(.1,min(mutation_r_s,.90)))
        IO_mutation(IO_array, Fitness_scores, mutation_rate = mutation_r_s,mutation_magnitude_mod = 1.0)
        
        previous_best_index = max_fitness_index
        previous_best_score = max_Fitness
        
        generation += 1
        
        # print(f"Mutation completed for Generation {generation}.")
        # with open(f"../genetic_alg/{SDN2}intermediate_files/logs/{generation}_log.txt","w") as f:
        #     f.write(f"Generation {generation} Fitness Scores:\n")
        #     for i, score in enumerate(Fitness_scores):
        #         f.write(f"Child {i}: {score}\n")
        

    final_max_fitness_index = np.nanargmax(Fitness_scores)
    print(f"Best overall child: Child {final_max_fitness_index} with fitness score: {Fitness_scores[final_max_fitness_index]}")



    






















if __name__ == "__main__":
    main()




#activity_gen_assign_populations.assign_jobs(output_file_name=f"../genetic_alg/static_files/Child_{child}_1.xml",job_assignments=job_array[child,:],real_edges=real_edges)
#            T_R_Command = f"activitygen --net-file ../data/sumo_network/greensboro.net.xml --stat-file ../genetic_alg/intermediate_files/Child_{child}_1.xml --output-file ../genetic_alg/intermediate_files/trips_routes/{child}_1_trips.rou.xml --seed 99"
#            os.system(f"{T_R_Command} > NUL 2>&1")
#            print(f"Generated trips file for Child {child}_1.xml")

#            R_Command = f"duarouter --net-file ../data/sumo_network/greensboro.net.xml --route-files ../genetic_alg/intermediate_files/trips_routes/{child}_1_trips.rou.xml --output-file ../genetic_alg/intermediate_files/routes/{child}_1_rou.xml --ignore-errors"
#            os.system(f"{R_Command} > NUL 2>&1")
#            print(f"Generated routes for Child {child}_1.xml")

#            tracker_assignment(child)

#            S_Command = f"sumo --net-file ../data/sumo_network/greensboro.net.xml --route-files ../genetic_alg/intermediate_files/routes/{child}_1_rou.xml --additional-files ../genetic_alg/intermediate_files/trackers/{child}_tracker.xml"
#            os.system(f"{S_Command} > NUL 2>&1")
#            print(f"SUMO simulation completed for Child {child}_1.xml")

#            none,none,correlations = correlation_test.test_correlation(output_path=f"../genetic_alg/intermediate_files/outputs/{child}_traffic.xml")
#            print(f"Correlation for Child {child}_1.xml: {correlations}")
#            if correlations == 'nan':
#                correlations = -1.0
