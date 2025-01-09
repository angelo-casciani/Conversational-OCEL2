import csv
import os
import random
import re


def read_event_log(filename):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', filename)
    with open(file_path, 'r') as file:
        return file.read()


def extract_traces_concept_names(log_content):
    trace_pattern = re.compile(r'<trace>.*?</trace>', re.DOTALL)
    event_pattern = re.compile(r'<event>.*?</event>', re.DOTALL)
    concept_name_pattern = re.compile(r'<string key="concept:name" value="(.*?)"/>')
    traces_list = []
    for trace_match in trace_pattern.findall(log_content):
        trace_content = []
        for event_match in event_pattern.findall(trace_match):
            concept_name_match = concept_name_pattern.search(event_match)
            if concept_name_match:
                concept_name = concept_name_match.group(1)
                trace_content.append(concept_name)
        if len(trace_content) >= 2:
            traces_list.append('; '.join(trace_content))
    traces_list = list(dict.fromkeys(traces_list))
    return traces_list


def extract_traces_with_attributes(log_content):
    trace_pattern = re.compile(r'<trace>.*?</trace>', re.DOTALL)
    event_pattern = re.compile(r'<event>.*?</event>', re.DOTALL)
    attribute_pattern = re.compile(r'key="(.*?)" value="(.*?)"/>')
    traces_list = []
    keys = {}
    activities = set()
    for trace_match in trace_pattern.findall(log_content):
        trace_content = []
        for event_match in event_pattern.findall(trace_match):
            attributes = []
            for attribute_match in attribute_pattern.findall(event_match):
                key, value = attribute_match
                #if key != "lifecycle:transition" and key != "time:timestamp":
                if key != "lifecycle:transition":
                    key_initial = ''.join([part[:2] for part in key.split(':')])
                    attributes.append(f'{key_initial}:{value}')
                    if key == 'concept:name':
                        activities.add(value)
                    if key_initial not in keys and key_initial != 'cona':
                        keys[key_initial] = key
            if attributes:
                trace_content.append(','.join(attributes))
        if len(trace_content) >= 2:
            traces_list.append('; '.join(trace_content))
    traces_list = list(dict.fromkeys(traces_list))
    return traces_list, keys, activities


def create_prefixes_with_attribute_last_values(traces, base, gap):
    prefixes_predictions = []
    for trace in traces:
        trace = trace.split('; ')
        indices = [i for i in range(base, len(trace), gap)]
        for index in indices:
            if index < len(trace):
                prefix = '; '.join(trace[:index]) + ';'
                prediction = trace[index].split('; ')[0].split(']')[0]
                prefix = prefix.rstrip(';') + ']'
                prefixes_predictions.append((prefix, prediction))
    return prefixes_predictions


def process_traces_with_last_attribute_values(traces):
    processed_traces = []
    for trace in traces:
        concept_name_trace = []
        attribute_values = {}
        trace = trace.split('; ')
        for event in trace:
            concept_name = event.split('cona:')[1].split(',')[0]
            concept_name_trace.append(concept_name)
            event_attributes = event.split(',')
            for attribute in event_attributes:
                attribute_key = attribute.split(':')[0]
                attribute_value = attribute.split(':')[1]
                if attribute_key != 'cona':
                    attribute_values[attribute_key] = attribute_value
        trace_attributes = str(concept_name_trace) + ' - Values: ' + str(attribute_values)
        processed_traces.append(trace_attributes)
    return processed_traces


def process_prefixes_prediction_with_last_attribute_values(prefix_prediction_pairs):
    processed_prefixes = []
    for pair in prefix_prediction_pairs:
        concept_name_trace = []
        attribute_values = {}
        prefix = pair[0].split('; ')
        prediction = pair[1].split('cona:')[1].split(',')[0]
        for event in prefix:
            concept_name = event.split('cona:')[1].split(',')[0]
            concept_name_trace.append(concept_name)
            event_attributes = event.split(',')
            for attribute in event_attributes:
                attribute_key = attribute.split(':')[0]
                attribute_value = attribute.split(':')[1]
                if attribute_key != 'cona':
                    attribute_values[attribute_key] = attribute_value
        proc_prefix_attributes = str(concept_name_trace) + ' - Values: ' + str(attribute_values)
        processed_prefixes.append((proc_prefix_attributes, prediction))
    return processed_prefixes


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set


def generate_csv_from_test_set(test_set, test_path):
    size = 300
    test_set = random.sample(test_set, size)
    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        for pair in test_set:
            csvwriter.writerow([pair[0], pair[1]])


"""def compute_log_stats(log_name):
    tree_content = read_event_log(log_name)
    traces = extract_traces(tree_content)
    print(f'Total number of traces: {len(traces)}')
    total_events = sum(len(trace) for trace in traces)
    print(f'Total number of events: {total_events}')


def main():
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets',
                                 f"sintetico-2-2var-1rel-1-nonrel.csv")
    content = read_event_log('sintetico-2-2var-1rel-1-nonrel.xes')
    traces, event_attributes, activities = extract_traces_with_attributes(content)
    print(activities)
    print(event_attributes)
    traces_to_store = process_traces_with_last_attribute_values(traces)
    test_set = generate_test_set(traces, 0.2)
    prefix_prediction = create_prefixes_with_attribute_last_values(test_set, base=1, gap=3)
    prefix_prediction_pairs = process_prefixes_prediction_with_last_attribute_values(prefix_prediction)
    generate_csv_from_test_set(test_set=prefix_prediction_pairs, test_path=test_set_path)


if __name__ == '__main__':
    main()
"""