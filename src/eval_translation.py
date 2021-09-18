import argparse
from pathlib import Path
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
from props.properties import penalized_logp, drd2, qed, similarity

from rdkit import RDLogger

# RDLogger.logger().setLevel(RDLogger.CRITICAL)
RDLogger.DisableLog("rdApp.info")
from rdkit.rdBase import BlockLogs

from rdkit import Chem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_list_path", type=str, default="")
    parser.add_argument("--task", type=str, default="logp04")
    hparams = parser.parse_args()

    if hparams.task == "logp04":
        score_func = lambda src, tgt: penalized_logp(tgt) - penalized_logp(src)
        similarity_thr = 0.4

    elif hparams.task == "logp06":
        score_func = lambda src, tgt: penalized_logp(tgt) - penalized_logp(src)
        similarity_thr = 0.6

    elif hparams.task == "drd2":
        score_func = lambda src, tgt: float(drd2(tgt) > 0.5)
        similarity_thr = 0.4

    elif hparams.task == "qed":
        score_func = lambda src, tgt: float(qed(tgt) > 0.9)
        similarity_thr = 0.4

    def batch_score_func(line):
        smiles_list = line.split(" ")
        src, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for tgt in tgt_list:
            try:
                score_list.append(score_func(src, tgt))
            except:
                score_list.append(None)

        return score_list

    def batch_similarity_func(line):
        smiles_list = line.split(" ")
        src, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for tgt in tgt_list:
            try:
                score_list.append(similarity(src, tgt))
            except:
                score_list.append(None)

        return score_list

    def batch_diversity_func(line):
        smiles_list = line.split(" ")
        _, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for smi0, smi1 in combinations(tgt_list, 2):
            try:
                score_list.append(1 - similarity(smi0, smi1))
            except:
                score_list.append(None)

        return score_list

    def batch_validity_func(line):
        smiles_list = line.split(" ")
        src, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for tgt in tgt_list:
            try:
                block = BlockLogs()
                mol = Chem.MolFromSmiles(tgt)
                tgt = Chem.MolToSmiles(mol)
                del block
                score_list.append(1.0)
            except:
                score_list.append(0.0)

        return score_list

    lines = Path(hparams.smiles_list_path).read_text(encoding="utf-8").splitlines()

    validity = Parallel(n_jobs=8)(delayed(batch_validity_func)(line) for line in lines)
    validity = np.array(validity, dtype=np.float)
    print(validity.mean())

    scores = Parallel(n_jobs=8)(delayed(batch_score_func)(line) for line in lines)
    similarities = Parallel(n_jobs=8)(delayed(batch_similarity_func)(line) for line in lines)

    scores = np.array(scores, dtype=np.float)
    similarities = np.array(similarities, dtype=np.float)

    thresholded_scores = scores.copy()
    thresholded_scores[similarities < similarity_thr] = np.nan
    thresholded_scores = np.nanmax(thresholded_scores, axis=1)
    print(np.nanmean(thresholded_scores), np.nanstd(thresholded_scores))

    diversities = Parallel(n_jobs=8)(delayed(batch_diversity_func)(line) for line in lines)
    diversities = np.nanmean(np.array(diversities, dtype=np.float), axis=1)
    print(np.nanmean(diversities), np.nanstd(diversities))
