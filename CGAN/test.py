import torch
import csv
import numpy as np
from generator import Generator
from get_flows import get_flows
from decode import decode_tensor
import random
from decimal import Decimal
from decode import decode_tensor

mappint_file_ids = 'output_ids.txt'
mapping_file_ports = 'output_ports.txt'
mapping_file_time = 'output_time_intervals.txt'


def decimal_to_10bit_binary(decimal_value):
    binary_str = bin(decimal_value)[2:]
    padded_binary_str = binary_str.zfill(10)
    binary_list = [int(bit) for bit in padded_binary_str]
    return binary_list

def convert_vector_to_binary(flow_vector):
    binary_representation = []
    for val in flow_vector:
        binary_representation.extend(decimal_to_10bit_binary(val.item()))
    return torch.tensor(binary_representation, dtype=torch.int)


def read_index_id_mapping(mapping_file):
    """
    从文件中读取索引映射回端口号的关系。
    :param mapping_file: str, 映射文件路径
    :return: dict, 索引到端口号的映射字典
    """
    index_id_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                index1 = int(parts[0])
                index2 = int(parts[1])
                index_id_mapping[index1] = index2
    return index_id_mapping


def read_index_timestamp_mapping(mapping_file):
    index_timestamp_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                index = int(parts[0])
                timestamp1 = float(parts[1])
                timestamp2 = float(parts[2])
                # 随机选择一个时间戳映射回原始时间戳
                mapped_timestamp = random.uniform(timestamp1, timestamp2)
                print(mapped_timestamp)
                index_timestamp_mapping[index] = mapped_timestamp
    return index_timestamp_mapping


def read_port_index_mapping(mapping_file):
    port_index_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                index1 = int(parts[0])
                index2 = int(parts[1])
                port_index_mapping[index2] = index1
    return port_index_mapping


def map_port_to_index(port, port_index_mapping):
    if port in port_index_mapping:
        return port_index_mapping[port]
    else:
        random_port_index = random.randint(1, 500)
        print(f"Warning: Port {port} is not in the mapping. Assigning random index {random_port_index}.")
        return random_port_index


def read_time_index_mapping(mapping_file):
    time_index_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                index = int(parts[0])
                time1 = Decimal(parts[1])
                time2 = Decimal(parts[2])
                time_index_mapping[(time1, time2)] = index
    return time_index_mapping


def map_time_to_index(time, time_index_mapping):
    for time_interval, index in time_index_mapping.items():
        if time_interval[0] <= time <= time_interval[1]:
            return index
    return 1



def ip_to_int_list(ip):
    return [int(part) for part in ip.split('.')]

def read_csv_to_array(csv_file):
    data = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            src_ip_parts = ip_to_int_list(row['src_ip'])
            dst_ip_parts = ip_to_int_list(row['dst_ip'])            
            src_port = int(row['src_port'])
            dst_port = int(row['dst_port'])
            protocol = int(row['protocol'])
            pkt_count = int(row['pkt_count'])
            timestamp = float(row['timestamp'])            
            data.append(src_ip_parts + dst_ip_parts + [src_port, dst_port, protocol, pkt_count, timestamp])

    return data

def process_condition_data(condition_datas):
    flow_vectors = []
    pkt_count_container = []
    for row in condition_datas:
        src_ip_parts = row[:4]
        dst_ip_parts = row[4:8]
        src_port, dst_port, protocol, pkt_count, start = row[8:]
        pkt_count_container.append(pkt_count)
        src_ip_parts = torch.tensor(src_ip_parts, dtype=torch.int)
        dst_ip_parts = torch.tensor(dst_ip_parts, dtype=torch.int)

        port_index_mapping = read_port_index_mapping(mapping_file_ports)
        src_port = map_port_to_index(src_port, port_index_mapping)
        src_port = torch.tensor(src_port, dtype=torch.int).unsqueeze(0)
        dst_port = map_port_to_index(dst_port, port_index_mapping)
        dst_port = torch.tensor(dst_port, dtype=torch.int).unsqueeze(0)

        protocol = torch.tensor(protocol, dtype=torch.int).unsqueeze(0)

        flow_vector = torch.cat([src_ip_parts, dst_ip_parts, src_port, dst_port, protocol])
        flow_vectors.append(flow_vector)

    return flow_vectors, pkt_count_container

def generate_pkt(flow_vectors, pkt_count_container):
    noisy_size = 1024 * 3
    output_dim = 60
    condition_dim = 110
    file_path = 'generator.pth'
    generator = load_generator(file_path, noisy_size, output_dim, condition_dim)
    generated_datas = []
    for i, flow_vector in enumerate(flow_vectors):
        flow_vector = convert_vector_to_binary(flow_vector)
        flow_vector = flow_vector.unsqueeze(0)
        pkt_count = pkt_count_container[i]
        for _ in range(pkt_count):
            noise = torch.randn(1, noisy_size)
            with torch.no_grad():
                generated_data = generator(noise, flow_vector)
                generated_data = generated_data.view(-1, output_dim)
                generated_data = torch.cat((flow_vector, generated_data), dim=1).reshape(170)
                generated_datas.append(generated_data)

    generated_datas = torch.stack(generated_datas, dim=0)
    generated_datas = generated_datas.reshape(-1, 170)
    return generated_datas


def postprecess(generated_datas):
    decoded_datas = decode_tensor(generated_datas, 10)
    num_samples = decoded_datas.size(0)
    for i in range(num_samples):
        id_idx = decoded_datas[i, 13].item()
        index_id_mapping = read_index_id_mapping(mappint_file_ids)
        if id_idx not in index_id_mapping:
            id_idx = 1
        id = index_id_mapping[id_idx]
        decoded_datas[i, 13] = id

        time_idx = decoded_datas[i, 15].item()
        index_time_mapping = read_index_timestamp_mapping(mapping_file_time)
        if time_idx not in index_time_mapping:
            time_idx = 1
        time = index_time_mapping[time_idx]
        decoded_datas[i, 15] = time

    return decoded_datas





def load_generator(checkpoint_path, noisy_size, output_dim, condition_dim):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    generator = Generator(noisy_size, output_dim, condition_dim)
    generator.load_state_dict(checkpoint)
    generator.eval()
    
    return generator


def tensor_to_csv(tensor, csv_filename):
    array = tensor.numpy()
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            src_ip = f"{int(row[0])}.{int(row[1])}.{int(row[2])}.{int(row[3])}"
            dst_ip = f"{int(row[4])}.{int(row[5])}.{int(row[6])}.{int(row[7])}"
            other_values = row[8:].tolist()
            writer.writerow([src_ip, dst_ip] + other_values)


def main():
    data = read_csv_to_array('generated_flows.csv')
    flow_vectors, pkt_count_container = process_condition_data(data)
    generated_datas = generate_pkt(flow_vectors, pkt_count_container)
    decoded_datas = postprecess(generated_datas)
    tensor_to_csv(decoded_datas, 'generated_pkt.csv')


if __name__ == "__main__":
    main()
