# -*- coding: utf-8 -*-
"""
build_drug_3d_coords_npy.py

Goal:
    Read response file -> collect required drug IDs -> find SMILES -> generate explicit-H 3D coordinates -> save one .npy file.

Output:
    npy file storing a dict:

    coord_dict[drug_key] = {
        "status": "ok",
        "smiles": smiles,
        "atoms": np.array([...], dtype=object),
        "coords3d": np.ndarray,      # [N, 3]
    }

    coord_dict[drug_key] = {
        "status": "failed",
        "smiles": smiles_or_empty,
        "error": error_message,
    }

Only failure cases kept:
    missing_in_smiles
    invalid_smiles
    embed_3d_failed
    timeout_over_xxs
    process_failed_no_message
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


# =========================================================
# Basic IO
# =========================================================
def read_table(path):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, header=0)
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=0)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def load_required_drugs(response_csv):
    df = read_table(response_csv)
    drug_col = df.columns[1]
    drugs = df[drug_col].dropna().astype(str).str.strip().unique().tolist()
    return sorted(set(drugs))


def load_smiles_dict(drug_csv):
    df = read_table(drug_csv)
    drug_col = df.columns[0]
    smiles_col = df.columns[1]

    df = df.dropna(subset=[drug_col, smiles_col]).copy()
    df[drug_col] = df[drug_col].astype(str).str.strip()
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[drug_col])

    return {
        str(row[drug_col]).strip(): str(row[smiles_col]).strip()
        for _, row in df.iterrows()
    }


# =========================================================
# 3D conformer generation
# =========================================================
def generate_3d_from_smiles(smiles, seed=2026, max_attempts=100):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid_smiles")

    mol_h = Chem.AddHs(mol)
    mol_h.UpdatePropertyCache(strict=False)

    params = AllChem.ETKDGv3()
    params.maxAttempts = int(max_attempts)
    params.ignoreSmoothingFailures = True
    params.randomSeed = int(seed)

    params.useRandomCoords = False
    ok = AllChem.EmbedMolecule(mol_h, params)

    if ok != 0:
        params.useRandomCoords = True
        ok = AllChem.EmbedMolecule(mol_h, params)

    if ok != 0:
        raise ValueError("embed_3d_failed")

    conf = mol_h.GetConformer()

    atoms = []
    coords = []
    for atom in mol_h.GetAtoms():
        idx = atom.GetIdx()
        p = conf.GetAtomPosition(idx)
        atoms.append(atom.GetSymbol())
        coords.append([p.x, p.y, p.z])

    return {
        "atoms": np.asarray(atoms, dtype=object),
        "coords3d": np.asarray(coords, dtype=np.float32),
    }


# =========================================================
# Timeout wrapper
# =========================================================
def _worker(queue, smiles, seed, max_attempts):
    try:
        result = generate_3d_from_smiles(
            smiles=smiles,
            seed=seed,
            max_attempts=max_attempts,
        )
        queue.put(("ok", result, ""))
    except Exception as e:
        queue.put(("failed", None, str(e)))


def build_with_timeout(smiles, seed=2026, max_attempts=100, timeout=30):
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)

    proc = ctx.Process(
        target=_worker,
        args=(queue, smiles, int(seed), int(max_attempts)),
    )
    proc.start()
    proc.join(float(timeout))

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, f"timeout_over_{timeout}s"

    if queue.empty():
        return None, "process_failed_no_message"

    status, payload, error = queue.get()
    if status != "ok":
        return None, error

    return payload, ""


# =========================================================
# Main build
# =========================================================
def build_drug_3d_npy(
    response_csv,
    drug_csv,
    out_npy,
    seed=2026,
    max_attempts=100,
    timeout=30,
):
    required_drugs = load_required_drugs(response_csv)
    smiles_dict = load_smiles_dict(drug_csv)

    print(f"Required drugs from response : {len(required_drugs)}")
    print(f"Drugs with SMILES            : {len(smiles_dict)}")
    print(f"Output npy                   : {out_npy}")
    print(f"Seed                         : {seed}")
    print(f"Max attempts                 : {max_attempts}")
    print(f"Timeout per drug             : {timeout}s")

    coord_dict = {}
    ok_count = 0
    fail_count = 0

    pbar = tqdm(required_drugs, desc="Generate 3D coords")
    for drug_key in pbar:
        pbar.set_postfix(drug=str(drug_key)[:20])

        if drug_key not in smiles_dict:
            coord_dict[drug_key] = {
                "status": "failed",
                "smiles": "",
                "error": "missing_in_smiles",
            }
            fail_count += 1
            continue

        smiles = smiles_dict[drug_key]
        result, error = build_with_timeout(
            smiles=smiles,
            seed=seed,
            max_attempts=max_attempts,
            timeout=timeout,
        )

        if result is None:
            coord_dict[drug_key] = {
                "status": "failed",
                "smiles": smiles,
                "error": error,
            }
            fail_count += 1
            continue

        coord_dict[drug_key] = {
            "status": "ok",
            "smiles": smiles,
            "atoms": result["atoms"],
            "coords3d": result["coords3d"],
        }
        ok_count += 1

    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, coord_dict, allow_pickle=True)

    print("\n========== 3D generation summary ==========")
    print(f"ok drugs     : {ok_count}")
    print(f"failed drugs : {fail_count}")
    print(f"saved to     : {out_npy}")
    print("==========================================\n")

    return coord_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--response_csv", default="/home/jzy/model/MolDr/CCLE/data/CCLE/CCLE_response.csv")
    parser.add_argument("--drug_csv", default="/home/jzy/model/MolDr/CCLE/data/CCLE/CCLE_Smiles.xlsx")
    parser.add_argument("--out_npy", default="/home/jzy/model/MolDr/CCLE/drug_3d_coords_H_all.npy")

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max_attempts", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    build_drug_3d_npy(
        response_csv=args.response_csv,
        drug_csv=args.drug_csv,
        out_npy=args.out_npy,
        seed=args.seed,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
