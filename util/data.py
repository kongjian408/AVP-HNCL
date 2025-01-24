import torch
import random
from util.seed import set_seed
import iFeatureOmegaCLI
from DDE import *
set_seed()


def load_data(data_path):
    avps = []
    nonavps = []

    with open(data_path, 'r', encoding='utf-8') as file:
        fasta_list = file.readlines()

    for flag in range(0, len(fasta_list), 2):
        header = fasta_list[flag].strip()
        sequence = fasta_list[flag + 1].strip()
        if 'pos' in header.lower():
            avps.append(sequence)
        else:
            nonavps.append(sequence)

    avps = list(avps)
    nonavps = list(nonavps)
    random.shuffle(avps)
    random.shuffle(nonavps)

    return avps, nonavps



def generate_features(input_path):

    AAC = iFeatureOmegaCLI.iProtein(input_path)
    AAC.get_descriptor("AAC")

    CKSAAGP = iFeatureOmegaCLI.iProtein(input_path)
    CKSAAGP.get_descriptor("CKSAAGP type 2")

    PAAC = iFeatureOmegaCLI.iProtein(input_path)
    PAAC.get_descriptor("PAAC")\

    QSOrder = iFeatureOmegaCLI.iProtein(input_path)
    QSOrder.get_descriptor("QSOrder")

    GTPC = iFeatureOmegaCLI.iProtein(input_path)
    GTPC.get_descriptor("GTPC type 2")

    DistancePair = iFeatureOmegaCLI.iProtein(input_path)
    DistancePair.get_descriptor("DistancePair")

    DPC = iFeatureOmegaCLI.iProtein(input_path)
    DPC.get_descriptor("DPC type 2")


    dde = feature_DDE(input_path)

    AAC.encodings = AAC.encodings.reset_index(drop=True)
    PAAC.encodings = PAAC.encodings.reset_index(drop=True)
    DistancePair.encodings = DistancePair.encodings.reset_index(drop=True)
    CKSAAGP.encodings = CKSAAGP.encodings.reset_index(drop=True)
    GTPC.encodings = GTPC.encodings.reset_index(drop=True)
    QSOrder.encodings = QSOrder.encodings.reset_index(drop=True)
    DPC.encodings = DPC.encodings.reset_index(drop=True)
    dde = dde.reset_index(drop=True)


    result = pd.concat([DistancePair.encodings,CKSAAGP.encodings, QSOrder.encodings, dde], axis=1)
    result.index = PAAC.encodings.index
    return result



z_scale_dict = {
    'A': [-1.56, -1.67, -1.30, 0.81, -0.21],
    'C': [0.12, 0.67, -2.05, -0.41, -0.09],
    'D': [1.06, 0.18, 1.23, -0.93, -0.89],
    'E': [0.88, 0.73, 1.26, -1.07, -0.74],
    'F': [-0.97, 0.27, -1.04, -0.25, 0.76],
    'G': [-1.22, -1.40, 1.23, -0.15, -1.13],
    'H': [0.64, -0.15, 1.05, -0.71, 0.94],
    'I': [-0.77, 0.84, -1.78, 1.15, -0.04],
    'K': [0.55, 1.68, 1.83, -0.80, -0.56],
    'L': [-0.72, 0.87, -1.41, 1.19, 0.23],
    'M': [-0.69, 0.62, -0.93, 0.45, 1.31],
    'N': [0.93, -0.56, 0.60, -0.60, 0.89],
    'P': [0.45, -0.09, 0.70, -1.05, 0.54],
    'Q': [0.90, 0.49, 0.83, -0.96, -0.19],
    'R': [1.84, 0.85, 1.41, -0.62, -1.07],
    'S': [0.20, -1.08, 0.24, -0.66, 0.48],
    'T': [0.32, -0.45, 0.00, -0.73, 0.53],
    'V': [-0.69, 1.30, -1.91, 1.15, -0.50],
    'W': [-0.39, 0.13, -0.73, 0.84, 2.10],
    'Y': [-1.47, 0.24, -0.14, 0.02, 1.65]
}



def sequence_to_zscale(sequence):
    zscale_representation = [z_scale_dict.get(aa, [0.0] * 5) for aa in sequence]
    return np.array(zscale_representation)

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_binary = {aa: np.eye(20)[i] for i, aa in enumerate(amino_acids)}

def sequence_to_binary(sequence):
    binary_representation = [aa_to_binary.get(aa, np.zeros(20)) for aa in sequence]
    return np.array(binary_representation)

def esm_encode(sequences, model, tokenizer, device, max_length, additional_features=None):
    tokens = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    with torch.no_grad():
        results = model(input_ids=input_ids, attention_mask=attention_mask)
    esm_representations = results.last_hidden_state.mean(dim=1)

    binary_features = [sequence_to_binary(seq).mean(axis=0) for seq in sequences]
    zscale_features = [sequence_to_zscale(seq).mean(axis=0) for seq in sequences]
    binary_tensor = torch.from_numpy(np.array(binary_features)).float().to(device)
    zscale_tensor = torch.from_numpy(np.array(zscale_features)).float().to(device)

    if additional_features is not None:
        additional_tensor = torch.from_numpy(additional_features).float().to(device)
        combined_representation = torch.cat((esm_representations, binary_tensor, zscale_tensor,additional_tensor), dim=1)
    else:
        combined_representation = torch.cat((esm_representations, binary_tensor, zscale_tensor), dim=1)

    return combined_representation
