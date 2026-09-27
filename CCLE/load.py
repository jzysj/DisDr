# -*- coding: utf-8 -*-
"""
load.py

Build four aligned cumulative 3D radius graphs for atom-consensus node separation:
    s0: 0 < d <= 2.0 A
    s1: 0 < d <= 2.5 A
    s2: 0 < d <= 3.0 A
    s3: 0 < d <= 3.5 A

The four views share nodes, node order, and atom features; only their adjacency structures differ.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


ATOM_LIST = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]


def read_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(path, header=0)
    else:
        frame = pd.read_csv(path, sep=None, engine="python", header=0)
    frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame.columns]
    return frame


def load_rnaseq(rnaseq_csv: str) -> Dict[str, np.ndarray]:
    frame = read_table(rnaseq_csv)
    cell_column = frame.columns[0]
    feature_columns = list(frame.columns[1:])

    cells = frame[cell_column].astype(str).str.strip().values
    matrix = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    matrix = matrix.values.astype(np.float32)
    return {cells[index]: matrix[index] for index in range(len(cells))}


def one_hot_unknown(value, choices: Sequence) -> list:
    if value not in choices:
        value = choices[-1]
    return [value == choice for choice in choices]


def total_hydrogens(atom: Chem.Atom) -> int:
    try:
        return int(atom.GetTotalNumHs(includeNeighbors=True))
    except TypeError:
        return int(atom.GetTotalNumHs())


def implicit_valence(atom: Chem.Atom) -> int:
    try:
        return int(atom.GetImplicitValence())
    except Exception:
        return 0


def atom_features(atom: Chem.Atom) -> np.ndarray:
    feature = (
        one_hot_unknown(atom.GetSymbol(), ATOM_LIST)
        + one_hot_unknown(int(atom.GetDegree()), list(range(11)))
        + one_hot_unknown(total_hydrogens(atom), list(range(11)))
        + one_hot_unknown(implicit_valence(atom), list(range(11)))
        + [bool(atom.GetIsAromatic())]
    )
    feature = np.asarray(feature, dtype=np.float32)
    feature_sum = float(feature.sum())
    if feature_sum > 0.0:
        feature = feature / feature_sum
    return feature


def build_atom_features(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("Invalid SMILES")
    molecule = Chem.AddHs(molecule)
    molecule.UpdatePropertyCache(strict=False)
    return np.asarray(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=np.float32,
    )


def pairwise_distance(coords3d: np.ndarray) -> np.ndarray:
    coords3d = np.asarray(coords3d, dtype=np.float32)
    difference = coords3d[:, None, :] - coords3d[None, :, :]
    return np.sqrt((difference * difference).sum(axis=-1)).astype(np.float32)


def build_radius_edges(distance_matrix: np.ndarray, radius: float) -> np.ndarray:
    mask = (distance_matrix > 0.0) & (distance_matrix <= float(radius))
    np.fill_diagonal(mask, False)
    source, target = np.where(mask)
    return np.vstack([source, target]).astype(np.int64)


def arrays_to_graph(
    atom_feature: np.ndarray,
    coords3d: np.ndarray,
    radii: Tuple[float, float, float, float],
) -> Data:
    distance_matrix = pairwise_distance(coords3d)
    graph = Data(x=torch.tensor(atom_feature, dtype=torch.float32))

    for scale, radius in enumerate(radii):
        edge_index = build_radius_edges(distance_matrix, radius)
        setattr(
            graph,
            f"edge_index_s{scale}",
            torch.tensor(edge_index, dtype=torch.long),
        )
    return graph


def build_drug_graphs(
    coordinate_records: dict,
    required_drugs: Iterable[str],
    radii: Tuple[float, float, float, float],
):
    drug_graphs = {}
    missing_drugs = []

    for drug_key in sorted(set(str(value).strip() for value in required_drugs)):
        record = coordinate_records.get(drug_key)
        if record is None or record.get("status") != "ok":
            missing_drugs.append(drug_key)
            continue

        try:
            coords3d = np.asarray(record["coords3d"], dtype=np.float32)
            atom_feature = build_atom_features(record["smiles"])
            if atom_feature.shape[0] != coords3d.shape[0]:
                missing_drugs.append(drug_key)
                continue

            drug_graphs[drug_key] = arrays_to_graph(
                atom_feature=atom_feature,
                coords3d=coords3d,
                radii=radii,
            )
        except Exception:
            missing_drugs.append(drug_key)

    return drug_graphs, missing_drugs


def load_dataset(
    response_csv: str,
    rnaseq_csv: str,
    coord_npy: str = "/home/jzy/model/MolDr/CTRP1/drug_3d_coords_H.npy",
    radii: Sequence[float] = (2.0, 2.5, 3.0, 3.5),
):
    radii = tuple(float(radius) for radius in radii)
    if len(radii) != 4:
        raise ValueError("Exactly four radii are required")
    if any(radii[index] >= radii[index + 1] for index in range(3)):
        raise ValueError("radii must be strictly increasing")

    response = read_table(response_csv)
    cell_column, drug_column, label_column = response.columns[:3]
    response = response.dropna(subset=[cell_column, drug_column, label_column]).copy()
    response[cell_column] = response[cell_column].astype(str).str.strip()
    response[drug_column] = response[drug_column].astype(str).str.strip()

    coordinate_records = np.load(coord_npy, allow_pickle=True).item()
    drug_graphs, missing_drugs = build_drug_graphs(
        coordinate_records=coordinate_records,
        required_drugs=response[drug_column].unique(),
        radii=radii,
    )

    response = response[response[drug_column].isin(drug_graphs)].copy()
    rna_dictionary = load_rnaseq(rnaseq_csv)

    rna_rows = []
    labels = []
    sample_drugs = []

    for _, row in response.iterrows():
        cell_key = str(row[cell_column]).strip()
        drug_key = str(row[drug_column]).strip()
        if cell_key not in rna_dictionary:
            continue
        rna_rows.append(rna_dictionary[cell_key])
        labels.append(float(row[label_column]))
        sample_drugs.append(drug_key)

    rna_matrix = np.vstack(rna_rows).astype(np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    sample_drugs = np.asarray(sample_drugs, dtype=object)
    atom_dim = int(next(iter(drug_graphs.values())).x.size(1))

    print("\n========== Data summary ==========")
    print(f"Matched samples : {len(labels)}")
    print(f"Valid drugs     : {len(drug_graphs)}")
    print(f"Missing drugs   : {len(missing_drugs)}")
    print(f"RNA dim         : {rna_matrix.shape[1]}")
    print(f"Atom dim        : {atom_dim}")
    print(f"Radii           : {radii}")
    print("==================================\n")

    return {
        "X_rna": rna_matrix,
        "y": labels,
        "sample_drugs": sample_drugs,
        "drug_graphs": drug_graphs,
        "atom_dim": atom_dim,
        "rna_dim": int(rna_matrix.shape[1]),
        "n_samples": int(len(labels)),
        "n_scales": 4,
        "radii": radii,
        "missing_drugs": missing_drugs,
    }
