from scapy.all import rdpcap
from collections import Counter
index_num = 254

def extract_time_intervals_from_pcap(file_path):
    packets = rdpcap(file_path)
    timestamps = [packet.time for packet in packets]

    min_time = min(timestamps)
    max_time = max(timestamps)
    interval_size = (max_time - min_time) / (index_num)

    index_mapping = []
    for timestamp in timestamps:
        interval_index = int((timestamp - min_time) / interval_size) + 1
        index = max(1, min(interval_index, index_num))
        index_mapping.append(index)

    return index_mapping, min_time, max_time

def save_to_file(data, output_file):
    with open(output_file, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

if __name__ == "__main__":
    file_path = 'caida_small.pcap'
    output_time_intervals_file = 'output_time_intervals.txt'
    index_mapping, min_time, max_time = extract_time_intervals_from_pcap(file_path)
    interval_size = (max_time - min_time) / index_num
    time_intervals_data = []
    for index in set(index_mapping):
        start_time = min_time + (index-1) * interval_size
        end_time = min_time + (index) * interval_size
        time_intervals_data.append(f"{index} {start_time} {end_time}")
    save_to_file(time_intervals_data, output_time_intervals_file)
    print(f"Time intervals have been saved to {output_time_intervals_file}")
