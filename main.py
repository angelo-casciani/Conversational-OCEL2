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
    resources = log['resource'].dropna().unique()
    return resources

if __name__ == "__main__":
    path = os.path.join("data", "ocel2-p2p.json")
    ocel = pm4py.read_ocel2(path)
    
    # Discover and view the DFG with the frequency annotation
    # ocdfg = pm4py.discover_ocdfg(ocel)
    # pm4py.view_ocdfg(ocdfg, format="svg")

    # Basic Statistics
    print(ocel)

    # Names of the attributes in the log
    attribute_names = pm4py.ocel_get_attribute_names(ocel)

    # Object Types
    object_types = pm4py.ocel_get_object_types(ocel)

    # Dictionary containing the set of activities for each object type 
    object_type_activities = pm4py.ocel_object_type_activities(ocel)

    # Number of related objects to the event for each event identifier and object type
    ocel_objects_ot_count = pm4py.ocel_objects_ot_count(ocel)

    # Temporal summary
    temporal_summary = pm4py.ocel_temporal_summary(ocel)
    # Object summary, a dataframe with the different objects in the log along with the activities of the events related to the object, the start/end timestamps of the lifecycle, the duration of the lifecycle and the other objects related to the given object in the interaction graph.
    objects_summary = pm4py.ocel_objects_summary(ocel)

    earliest, latest, total = compute_times(temporal_summary)
    print("Earliest event:", earliest)
    print("Latest event:", latest)
    print("Total duration:", total)


    # Operations on a flattened log for object type
    flattened_log = pm4py.ocel_flattening(ocel, "invoice receipt")
    resources = get_resources_flat(flattened_log)
    print(flattened_log)
    with open('test.txt', 'w') as file:
        file.write(flattened_log.to_string())
    
    """
    with open('test.txt', 'w') as file:
        file.write(objects_summary.to_string())

    Flattened Log Columns: 'ocel:eid', 'time:timestamp', 'concept:name', 'lifecycle', 'resource', 'case:concept:name', 'case:ocel:type',  ... Case for each Event Type
    """