import re
import math
from util.seed import set_seed
import pandas as pd
import numpy as np

set_seed()

def DDE(fastas, **kw):
	AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'K', 'L', 'M', 'N', 'P', 'Q',
          'R', 'S', 'T', 'V', 'W', 'Y']

	myCodons = {
		'A': 4,
		'C': 2,
		'D': 2,
		'E': 2,
		'F': 2,
		'G': 4,
		'H': 2,
		'I': 3,
		'K': 2,
		'L': 6,
		'M': 1,
		'N': 2,
		'P': 4,
		'Q': 2,
		'R': 6,
		'S': 6,
		'T': 4,
		'V': 4,
		'W': 1,
		'Y': 2
	}

	encodings = []
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	header = ['#'] + diPeptides
	encodings.append(header)

	myTM = []
	for pair in diPeptides:
		myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]

		myTV = []
		for j in range(len(myTM)):
			myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

		for j in range(len(tmpCode)):
			tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

		code = code + tmpCode
		encodings.append(code)
	return encodings


def feature_DDE(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    fasta_list = np.array(f.readlines())
    aa_feature_list = []
    for flag in range(0, len(fasta_list), 2):
        fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]
        dpc_output = DDE(fasta_str)
        dpc_output[1].remove(dpc_output[1][0])
        dpc_feature = dpc_output[1][:]
        aa_feature_list.append(dpc_feature)
    aa_feature_list = pd.DataFrame(aa_feature_list)
    aa_feature_list = aa_feature_list.iloc[:,:]
    coloumnname = []
    for i in range(len(aa_feature_list.columns)):
        x = 'DDE'+str(i+1)
        coloumnname.append(x)
    aa_feature_list.columns = coloumnname
    return aa_feature_list

