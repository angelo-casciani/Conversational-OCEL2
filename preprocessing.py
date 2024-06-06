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


def get_resource_calendar(log):
    log = log[log['lifecycle'] == 'complete']
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])

    log['weekday'] = log['time:timestamp'].dt.weekday
    log['dayname'] = log['time:timestamp'].dt.day_name()
    log['time'] = log['time:timestamp'].dt.time
    min_times = log.groupby(['resource', log['weekday'], log['dayname']])['time'].min()
    max_times = log.groupby(['resource', log['weekday'], log['dayname']])['time'].max()
    resource_times = pd.DataFrame({'min_time': min_times, 'max_time': max_times})

    return resource_times


def compute_stats_numeric_attributes(log, num_attribute):
    attribute = f'case:{num_attribute}'
    log[attribute] = pd.to_numeric(log[attribute], errors='coerce')

    log = log.dropna(subset=[attribute])
    if log.empty:
        return None, None, None

    minimum_row = log.loc[log[attribute].idxmin()]
    maximum_row = log.loc[log[attribute].idxmax()]
    average = log[attribute].mean()

    return minimum_row, maximum_row, average


def compute_stats_string_attributes(log, str_attribute):
    attribute = f'case:{str_attribute}'
    unique_values = log[attribute].unique().tolist()

    return unique_values


def compute_stats_date_attributes(log, date_attribute):
    attribute = f'case:{date_attribute}'
    log[attribute] = pd.to_datetime(log[attribute], errors='coerce')

    log = log.dropna(subset=[attribute])
    if log.empty:
        return None, None

    minimum_row = log.loc[log[attribute].idxmin()]
    maximum_row = log.loc[log[attribute].idxmax()]

    return minimum_row, maximum_row


def empty_execution_data(exe_dir):
    files = os.listdir(exe_dir)

    for f in files:
        if f.endswith('.txt'):
            file_path = os.path.join(exe_dir, f)
            os.remove(file_path)

    print("All execution files have been deleted.")


def format_df_value(val, name):
    return f'{name}: {val} | '

def delete_files_no_info():
    files = os.listdir(os.path.join('data', 'execution'))
    criteria = ('date_attributes', 'string_attributes', 'numeric_attributes')

    for file in files:
        if file.startswith(criteria) and file.endswith('.txt'):
            file_path = os.path.join('data', 'execution', file)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) <= 1:
                        f.close()
                        os.remove(file_path)
                        print(f'Deleted file: {file_path}')
            except PermissionError:
                print(f'Cannot delete file: {file_path} as it is being used by another process')
            except Exception as e:
                print(f'An error occurred: {e}')


if __name__ == "__main__":
    directory = os.path.join('data', 'execution')
    if not os.path.exists(directory):
        os.makedirs(directory)
    empty_execution_data(directory)

    path = os.path.join("data", "ocel2-p2p.json")
    ocel = pm4py.read_ocel2(path)

    # Basic Statistics
    filename = os.path.join('data', 'execution', 'basic_stats.txt')
    with open(filename, 'w') as file:
        file.write(f'OCEL2.0 basic statistics.\n')
        print(ocel, file=file)
    with open(filename, 'r') as file:
        lines = file.readlines()
    with open(filename, 'w') as file:
        file.writelines(lines[:-1])

    # Names of the attributes in the log
    attribute_names = pm4py.ocel_get_attribute_names(ocel)
    attributes_as_string = ', '.join(attribute_names)
    with open(os.path.join('data', 'execution', 'attribute_names.txt'), 'w') as file:
        file.write(f'OCEL2.0 object types, a list containing the attribute names.\n')
        file.write(attributes_as_string)

    # Object Types
    object_types = pm4py.ocel_get_object_types(ocel)
    obj_as_string = ', '.join(object_types)
    with open(os.path.join('data', 'execution', 'object_types.txt'), 'w') as file:
        file.write(f'OCEL2.0 object types: a list containing the object types.\n')
        file.write(obj_as_string)

    # Dictionary containing the set of activities for each object type
    object_type_activities = pm4py.ocel_object_type_activities(ocel)
    dict_serializable = {key: list(value) for key, value in object_type_activities.items()}
    dict_as_string = json.dumps(dict_serializable)
    with open(os.path.join('data', 'execution', 'object_type_activities.txt'), 'w') as file:
        file.write(
            f'OCEL2.0 object type activities: the set of activities for each object type.\n')
        file.write(dict_as_string)

    # Number of related objects to the event for each event identifier and object type
    ocel_objects_ot_count = pm4py.ocel_objects_ot_count(ocel)
    dict_as_string = json.dumps(ocel_objects_ot_count)
    with open(os.path.join('data', 'execution', 'to_chunk', 'objects_ot_count.txt'), 'w') as file:
        #file.write(f'OCEL2.0 objects ot count: number of related objects to the event for each event identifier and object type.\n')
        file.write(dict_as_string)

    # Temporal info
    temporal_summary = pm4py.ocel_temporal_summary(ocel)
    formatted_columns = {col: lambda x, name=col: format_df_value(x, name) for col in temporal_summary.columns}
    with open(os.path.join('data', 'execution', 'to_chunk', 'temporal_summary.txt'), 'w') as file:
        file.write(f'OCEL2.0 temporal summary: the events happening in a specific timestamp and the objects on which they operate.\n')
        file.write(temporal_summary.to_string(formatters=formatted_columns, index=False))

    # Temporal Process Wide Info
    earliest, latest, total = compute_times(temporal_summary)
    with open(os.path.join('data', 'execution', 'temporal_process_info.txt'), 'w') as file:
        file.write(f'Earliest event timestamp: {earliest}\n')
        file.write(f'Latest event timestamp: {latest}\n')
        file.write(f'Total duration: {total}\n')

    # Object summary
    objects_summary = pm4py.ocel_objects_summary(ocel)
    formatted_columns = {col: lambda x, name=col: format_df_value(x, name) for col in objects_summary.columns}
    with open(os.path.join('data', 'execution', 'to_chunk', 'object_summary.txt'), 'w') as file:
        file.write(
            f'OCEL2.0 object summary: info on the objects with the events related to the object, the start/end timestamps and duration of the object lifecycle, and the other objects interacting with the given object.\n')
        file.write(objects_summary.to_string(formatters=formatted_columns, index=False))

    # Operations on a flattened log for each object type
    with open(os.path.join('data', 'execution', 'object_types.txt'), 'r') as file:
        next(file)
        second_line = next(file)
        obj_types_list = second_line.split(",")
        obj_types_list = [item.strip() for item in obj_types_list]

    # Attributes specific for the selected log
    numeric_attributes = ['Amount (DMBTR)', 'Credit Amount (BSEG-WRBTR)', 'Debit Amount (BSEG-DMBTR)',
                          'Net Price (EKPO-NETPR)', 'Quantity (EKPO-MENGE)']
    date_attributes = ['Delivery Date (EKPO-BEDAT)', 'Posting Date (MKPF-BLDAT)']
    string_attributes = ['Document Type (EKKO-BSART)', 'Material (EKPO-MATNR)', 'Movement Type (MSEG-BWART)',
                         'Payment Block (RSEG-ZLSPR)', 'Payment Method (ZLSCH)',
                         'Plant (EKPO-WERKS)', 'Invoice Receipt (MSEG-WEAHR)', 'Vendor (EBAN-LIFNR)',
                         'Vendor (EKKO-LIFNR)',
                         'Storage Location (EKPO-LGORT)', 'RFQ Type (EBAN-BSART)', 'Release Indicator (EBAN-FRGZU)',
                         'Release Status (EKKO-FRGZU)', 'Purchasing Organization (EBAN-EKORG)',
                         'Purchasing Organization (EKKO-EKORG)', 'Purchasing Group (EBAN-EKGRP)',
                         'Purchasing Group (EKKO-EKGRP)']

    for obj_type in obj_types_list:
        flattened_log = pm4py.ocel_flattening(ocel, obj_type)
        # Resources Stats
        resources = get_resources_flat(flattened_log)
        calendar = get_resource_calendar(flattened_log)
        filename = f'resources_{obj_type.replace(" ", "_")}.txt'
        with open(os.path.join('data', 'execution', filename), 'w') as file:
            # file.write(flattened_log.to_string())
            file.write(
                f'Resources and their working calendar for the OCEL2.0 event log flattened on the object type "{obj_type}".\n')
            file.write(f'The resources that manipulates an object of type "{obj_type}" are: {resources}.\n')
            file.write(f'The working calendar of these resources is as follows:\n {calendar}\n')
        # Other Attributes Stats
        filename = f'numeric_attributes_stats_{obj_type.replace(" ", "_")}.txt'
        with open(os.path.join('data', 'execution', filename), 'w') as file:
            file.write(
                f'Statistics about the numeric attribute values for the OCEL2.0 event log flattened on the object type "{obj_type}".\n')
            for num_attr in numeric_attributes:
                min_val_row, max_val_row, avg = compute_stats_numeric_attributes(flattened_log, num_attr)
                if isinstance(min_val_row, pd.Series) and isinstance(max_val_row, pd.Series):
                    min_val = min_val_row[f'case:{num_attr}']
                    max_val = max_val_row[f'case:{num_attr}']
                    min_case_name = min_val_row['case:concept:name']
                    max_case_name = max_val_row['case:concept:name']
                    min_ocel_eid = min_val_row['ocel:eid']
                    max_ocel_eid = max_val_row['ocel:eid']
                    avg = round(avg, 2)
                    file.write(f'"{num_attr}"\nMinimum: {min_val} with object {min_case_name} in event {min_ocel_eid} - Maximum: {max_val} with object {max_case_name} in event {max_ocel_eid}  - Average: {avg}\n')
        filename = f'string_attributes_stats_{obj_type.replace(" ", "_")}.txt'
        with open(os.path.join('data', 'execution', filename), 'w') as file:
            file.write(
                f'Statistics about the string attribute values for the OCEL2.0 event log flattened on the object type "{obj_type}".\n')
            for str_attr in string_attributes:
                values = compute_stats_string_attributes(flattened_log, str_attr)
                if not (pd.isna(values).any()):
                    file.write(f'"{str_attr}"\nValues: {values}\n')
        filename = f'date_attributes_stats_{obj_type.replace(" ", "_")}.txt'
        with open(os.path.join('data', 'execution', filename), 'w') as file:
            file.write(
                f'Statistics about the date attribute values for the OCEL2.0 event log flattened on the object type "{obj_type}".\n')
            for date_attr in date_attributes:
                min_val_row, max_val_row = compute_stats_date_attributes(flattened_log, date_attr)
                if isinstance(min_val_row, pd.Series) and isinstance(max_val_row, pd.Series):
                    min_val = min_val_row[f'case:{date_attr}']
                    max_val = max_val_row[f'case:{date_attr}']
                    min_case_name = min_val_row['case:concept:name']
                    max_case_name = max_val_row['case:concept:name']
                    min_ocel_eid = min_val_row['ocel:eid']
                    max_ocel_eid = max_val_row['ocel:eid']
                    file.write(f'"{date_attr}"\nMinimum Date: {min_val} with object {min_case_name} in event {min_ocel_eid} - Maximum Date: {max_val} with object {max_case_name} in event {max_ocel_eid}\n')

    delete_files_no_info()
