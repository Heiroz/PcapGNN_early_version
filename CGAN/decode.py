import torch

def decode_binary_vector_to_integers(binary_vector, bit_length=256):
    decoded_values = []
    for i in range(0, binary_vector.size(1), bit_length):
        binary_str = ''.join(str(int(bit)) for bit in binary_vector[0, i:i+bit_length] if str(int(bit)) in ['0', '1'])
        if len(binary_str) > 0:
            decoded_values.append(int(binary_str, 2))
        else:
            decoded_values.append(0)
    return torch.tensor(decoded_values, dtype=torch.float)

def decode_tensor(encoded_tensor, bit_length=256):
    num_samples = encoded_tensor.size(0)
    dim = encoded_tensor.size(1) // bit_length
    decoded_tensor = torch.zeros(num_samples, dim, dtype=torch.float)
    
    for i in range(num_samples):
        decoded_tensor[i] = decode_binary_vector_to_integers(encoded_tensor[i].unsqueeze(0), bit_length)
    
    return decoded_tensor