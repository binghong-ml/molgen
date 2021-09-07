from tqdm import tqdm

import os, sys
import networkx as nx
from joblib import Parallel, delayed

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from moses.metrics import internal_diversity

import torch

def _raw_plogp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
    except:
        return None
    
    LOGP_MEAN = 2.4570965532649507
    LOGP_STD = 1.4339810636722639
    SASCORE_MEAN = 3.0508333383104556
    SASCORE_STD = 0.8327034846660627
    CYCLESCORE_MEAN = 0.048152237188108474
    CYCLESCORE_STD = 0.2860582871837183
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        log_p = Descriptors.MolLogP(mol)
        sa_score = sascorer.calculateScore(mol)
    except:
        return None
        
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest_ring_size = max([len(j) for j in cycle_list]) if cycle_list else 0
    cycle_score = max(largest_ring_size - 6, 0)

    log_p = (log_p - LOGP_MEAN) / LOGP_STD
    sa_score = (sa_score - SASCORE_MEAN) / SASCORE_STD
    cycle_score = (cycle_score - CYCLESCORE_MEAN) / CYCLESCORE_STD

    return log_p - sa_score - cycle_score
    
def _raw_logp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
        return Descriptors.MolLogP(mol)
    except:
        return None

def _raw_molwt(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
        return Descriptors.MolWt(mol)
    except:
        return None

def _raw_qed(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol)
        return Descriptors.qed(mol)
    except:
        return None

from rdkit import DataStructs

def _raw_plogp_improvement(smiles0, smiles1, threshold):
    try:
        mol0, mol1 = Chem.MolFromSmiles(smiles0), Chem.MolFromSmiles(smiles1)
        _, _ = Chem.MolToSmiles(mol0), Chem.MolToSmiles(mol1)
    except:
        return None

    fp0, fp1 = Chem.RDKFingerprint(mol0), Chem.RDKFingerprint(mol1)
    similarity = DataStructs.FingerprintSimilarity(fp0, fp1)
    if similarity < threshold:
        return None
        
    plogp0, plogp1 = _raw_plogp(smiles0), _raw_plogp(smiles1)
    return plogp1 - plogp0
    