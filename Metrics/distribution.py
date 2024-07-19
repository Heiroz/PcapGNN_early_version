from scapy.all import IP, rdpcap
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def extract_packet_features(pcap_file):
    packets = rdpcap(pcap_file)
    ttl_values = []
    id_values = []
    tos_values = []
    flag_values = []
    len_pkt_values = []
    timestamp_values = []

    for pkt in packets:
        if pkt.haslayer(IP):
            ip_pkt = pkt.getlayer(IP)
            ttl_values.append(ip_pkt.ttl)
            id_values.append(ip_pkt.id)
            tos_values.append(ip_pkt.tos)
            flag_values.append(str(ip_pkt.flags))  # Convert to string
            len_pkt_values.append(len(pkt))
            timestamp_values.append(pkt.time)

    return ttl_values, id_values, tos_values, flag_values, len_pkt_values, timestamp_values

def smooth_curve(data, window_size=10):
    if len(data) < window_size:
        window_size = len(data)
    weights = np.repeat(1.0, window_size) / window_size
    smoothed = np.convolve(data, weights, 'valid')
    return smoothed

def plot_continuous_distribution(data, title, x_label, smooth=True, window_size=10, save_path=None):
    plt.figure(figsize=(10, 6))
    if smooth:
        smoothed_data = smooth_curve(data, window_size)
        plt.plot(smoothed_data, label='Smoothed Curve', linewidth=2, color='blue')
    else:
        plt.plot(data, label='Original Data', linewidth=1, color='gray', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Packet Count', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_discrete_distribution(data, title, x_label, save_path=None):
    plt.figure(figsize=(10, 6))
    counter = Counter(data)
    x = list(counter.keys())
    y = list(counter.values())
    plt.bar(x, y, edgecolor='black', color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Packet Count', fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main(pcap_file):
    ttl_values, id_values, tos_values, flag_values, len_pkt_values, timestamp_values = extract_packet_features(pcap_file)
    
    plot_continuous_distribution(ttl_values, 'TTL Distribution', 'Packet Index', smooth=True, save_path='ttl_distribution_smooth.png')
    plot_discrete_distribution(id_values, 'ID Distribution', 'ID Value', save_path='id_distribution.png')
    plot_discrete_distribution(tos_values, 'TOS Distribution', 'TOS Value', save_path='tos_distribution.png')
    plot_discrete_distribution(flag_values, 'Flag Distribution', 'Flag Value', save_path='flag_distribution.png')
    plot_continuous_distribution(len_pkt_values, 'Packet Length Distribution', 'Packet Length', smooth=True, save_path='packet_length_distribution_smooth.png')
    plot_continuous_distribution(timestamp_values, 'Timestamp Distribution', 'Packet Index', smooth=True, save_path='timestamp_distribution_smooth.png')

if __name__ == '__main__':
    pcap_file = 'caida_small.pcap'
    main(pcap_file)
