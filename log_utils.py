import json
import os
import pm4py
import pandas as pd


def compute_times(temp_summ_df):
    temp_summ_df['ocel:timestamp'] = pd.to_datetime(temp_summ_df['ocel:timestamp'])

    earliest_event = temp_summ_df['ocel:timestamp'].min()
    latest_event = temp_summ_df['ocel:timestamp'].max()
    total_duration = latest_event - earliest_event

    return earliest_event, latest_event, total_duration


# Get the resources for a flattened event log as a dataframe
def get_resources_flat(log):
    res = log['resource'].dropna().unique()
    return res


def empty_execution_data(exe_dir):
    files = os.listdir(exe_dir)

    for f in files:
        if f.endswith('.txt'):
            file_path = os.path.join(exe_dir, f)
            os.remove(file_path)

    print("All execution files have been deleted.")


if __name__ == "__main__":
    directory = os.path.join('data', 'execution')
    if not os.path.exists(directory):
        os.makedirs(directory)
    empty_execution_data(directory)

    path = os.path.join("data", "ocel2-p2p.json")
    ocel = pm4py.read_ocel2(path)
    
    # Discover and view the DFG with the frequency annotation
    # ocdfg = pm4py.discover_ocdfg(ocel)
    # pm4py.view_ocdfg(ocdfg, format="svg")

    # Basic Statistics
    with open(os.path.join('data', 'execution', 'basic_stats.txt'), 'w') as file:
        file.write(f'OCEL2.0 basic statistics.\n')
        print(ocel, file=file)
      
    # Names of the attributes in the log
    attribute_names = pm4py.ocel_get_attribute_names(ocel)
    attributes_as_string = ', '.join(attribute_names)
    with open(os.path.join('data', 'execution', 'attribute_names.txt'), 'w') as file:
        file.write(f'OCEL2.0 object types, a list containing the attribute names.\n')
        file.write(attributes_as_string)

    # Object Types
    object_types = pm4py.ocel_get_object_types(ocel)
    obj_as_string = ', '.join(attribute_names)
    with open(os.path.join('data', 'execution', 'object_types.txt'), 'w') as file:
        file.write(f'OCEL2.0 object types, a list containing the object types.\n')
        file.write(obj_as_string)

    # Dictionary containing the set of activities for each object type 
    object_type_activities = pm4py.ocel_object_type_activities(ocel)
    dict_serializable = {key: list(value) for key, value in object_type_activities.items()}
    dict_as_string = json.dumps(dict_serializable)
    with open(os.path.join('data', 'execution', 'object_type_activities.txt'), 'w') as file:
        file.write(f'OCEL2.0 object type activities, a dictionary containing the set of activities for each object type.\n')
        file.write(dict_as_string)

    # Number of related objects to the event for each event identifier and object type
    ocel_objects_ot_count = pm4py.ocel_objects_ot_count(ocel)
    dict_as_string = json.dumps(ocel_objects_ot_count)
    with open(os.path.join('data', 'execution', 'objects_ot_count.txt'), 'w') as file:
        file.write(f'OCEL2.0 objects ot count, number of related objects to the event for each event identifier and object type.\n')
        file.write(dict_as_string)

    # Temporal info
    temporal_summary = pm4py.ocel_temporal_summary(ocel)
    with open(os.path.join('data', 'execution', 'temporal_summary.txt'), 'w') as file:
        file.write(f'OCEL2.0 temporal summary.\n')
        file.write(temporal_summary.to_string())

    # Temporal Process Wide Info
    earliest, latest, total = compute_times(temporal_summary)
    with open(os.path.join('data', 'execution', 'temporal_process_info.txt'), 'w') as file:
        file.write(f'Earliest event timestamp: {earliest}\n')
        file.write(f'Latest event timestamp: {latest}\n')
        file.write(f'Total duration: {total}\n')

    # Object summary
    objects_summary = pm4py.ocel_objects_summary(ocel)
    with open(os.path.join('data', 'execution', 'object_summary.txt'), 'w') as file:
        file.write(f'OCEL2.0 object summary, a dataframe with the different objects in the log along with the activities of the events related to the object, the start/end timestamps of the lifecycle, the duration of the lifecycle and the other objects related to the given object in the interaction graph.\n')
        file.write(objects_summary.to_string())

    # Operations on a flattened log for object type
    obj_type = 'invoice receipt'
    flattened_log = pm4py.ocel_flattening(ocel, obj_type)
    resources = get_resources_flat(flattened_log)
    filename = f'flattened_log_{obj_type}.txt' 
    with open(os.path.join('data', 'execution', filename), 'w') as file:
        file.write(f'OCEL2.0 event log flattened on the object type "{obj_type}".\n')
        file.write(flattened_log.to_string())