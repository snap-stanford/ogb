"""PCQM4Mv2 compatibility checks against graphs from RDKit 2024.03.6."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from ogb.utils import smiles2graph
from ogb.utils.pcqm4mv2 import _LEGACY_SMILES, smiles2graph_pcqm4mv2


FIXTURE = json.loads(
    (Path(__file__).parent / 'data/pcqm4mv2_legacy_graphs.json').read_text())


def graph_digest(graph):
    values = {key: value.tolist() if isinstance(value, np.ndarray) else value
              for key, value in graph.items()}
    encoded = json.dumps(values, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_compatibility_is_limited_to_reported_records():
    assert len(FIXTURE['records']) == 39
    assert {row['smiles'] for row in FIXTURE['records']} == _LEGACY_SMILES


@pytest.mark.parametrize('record', FIXTURE['records'],
                         ids=lambda record: str(record['index']))
@pytest.mark.parametrize('remove_hs', [True, False])
@pytest.mark.parametrize('reorder', [False, True])
def test_legacy_graph_features(record, remove_hs, reorder):
    graph = smiles2graph_pcqm4mv2(record['smiles'], remove_hs, reorder)
    key = 'removeHs={},reorder_atoms={}'.format(remove_hs, reorder)
    # Covers every node feature, edge feature, atom index and edge ordering,
    # including legacy radical counts and optional explicit hydrogens.
    assert graph_digest(graph) == record['sha256'][key]
    assert graph['node_feat'].dtype == np.int64
    assert graph['edge_feat'].dtype == np.int64
    assert graph['edge_index'].dtype == np.int64


@pytest.mark.parametrize('smiles', [
    'CCO', 'c1ccccc1', 'C[C@H](O)C(=O)O', 'F/C=C/F',
    '[Na+].[Cl-]', '[CH3]', '[H]C([H])([H])[H]', '[SiH4]',
    'C[Si](C)(C)C', 'OP(=O)(O)O', 'C1CCCCC1', '[He]',
])
@pytest.mark.parametrize('remove_hs', [True, False])
@pytest.mark.parametrize('reorder', [False, True])
def test_ordinary_graphs_are_unchanged(smiles, remove_hs, reorder):
    expected = smiles2graph(smiles, remove_hs, reorder)
    actual = smiles2graph_pcqm4mv2(smiles, remove_hs, reorder)
    assert graph_digest(actual) == graph_digest(expected)


@pytest.mark.parametrize('smiles', [
    'not-a-smiles', 'C1CC', 'C(C)(C)(C)(C)C', 'CN(C)(C)C',
    # The allowlist is exact, not a blanket exception for Si/P valence.
    'O[Si]123O[Si]3(O1)(O2)O.C(C)(C)(C)(C)C',
    'O[Si]123O[Si]3(O1)(O2)O.',
])
def test_unrecognized_invalid_smiles_stay_rejected(smiles):
    with pytest.raises(AttributeError):
        smiles2graph(smiles)
    with pytest.raises(AttributeError):
        smiles2graph_pcqm4mv2(smiles)


def test_generic_converter_keeps_rdkit_valence_rules():
    smiles = FIXTURE['records'][0]['smiles']
    if Chem.MolFromSmiles(smiles) is None:
        with pytest.raises(AttributeError):
            smiles2graph(smiles)
    else:
        # The strict parser accepted this molecule before RDKit 2024.09.
        assert graph_digest(smiles2graph(smiles)) == graph_digest(
            smiles2graph_pcqm4mv2(smiles))


def test_unlisted_legacy_valence_follows_strict_parser():
    smiles = 'C[Si](C)(C)(C)C'
    assert smiles not in _LEGACY_SMILES
    if Chem.MolFromSmiles(smiles) is None:
        with pytest.raises(AttributeError):
            smiles2graph_pcqm4mv2(smiles)
    else:
        assert graph_digest(smiles2graph_pcqm4mv2(smiles)) == graph_digest(
            smiles2graph(smiles))


def test_pcqm_rows_labels_and_custom_converter_are_preserved(tmp_path):
    torch = pytest.importorskip('torch')
    pandas = pytest.importorskip('pandas')
    from ogb.lsc.pcqm4mv2 import PCQM4Mv2Dataset

    folder = tmp_path / 'pcqm4m-v2'
    (folder / 'raw').mkdir(parents=True)
    (folder / 'processed').mkdir()
    (folder / 'RELEASE_v1.txt').touch()
    smiles = ['CCO', FIXTURE['records'][0]['smiles'], 'c1ccccc1', '[He]']
    labels = [0.25, 1.5, float('nan'), float('nan')]
    pandas.DataFrame({'smiles': smiles, 'homolumogap': labels}).to_csv(
        folder / 'raw/data.csv.gz', index=False)
    # The official archive stores NumPy split indices, not Torch tensors.
    splits = {'train': np.array([1]), 'valid': np.array([0]),
              'test-dev': np.array([2]), 'test-challenge': np.array([3])}
    torch.save(splits, folder / 'split_dict.pt')

    for _ in range(2):
        dataset = PCQM4Mv2Dataset(root=str(tmp_path))
        assert len(dataset) == len(smiles)
        np.testing.assert_equal(dataset.labels, labels)
        for i, text in enumerate(smiles):
            assert graph_digest(dataset[i][0]) == graph_digest(
                smiles2graph_pcqm4mv2(text))
        for split, indices in dataset.get_idx_split().items():
            np.testing.assert_array_equal(indices, splits[split])

    # A supplied converter remains authoritative; it sees every row in order.
    (folder / 'processed/data_processed').unlink()
    seen = []

    def custom(text):
        seen.append(text)
        return smiles2graph('CC')

    custom_dataset = PCQM4Mv2Dataset(root=str(tmp_path), smiles2graph=custom)
    assert seen == smiles
    assert len(custom_dataset) == len(smiles)
    assert all(graph_digest(g) == graph_digest(smiles2graph('CC'))
               for g in custom_dataset.graphs)


def test_pyg_pcqm_process_and_reload_preserve_rows(tmp_path):
    torch = pytest.importorskip('torch')
    pandas = pytest.importorskip('pandas')
    pytest.importorskip('torch_geometric')
    from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset

    folder = tmp_path / 'pcqm4m-v2'
    (folder / 'raw').mkdir(parents=True)
    (folder / 'RELEASE_v1.txt').touch()
    smiles = ['CCO', FIXTURE['records'][0]['smiles'], 'c1ccccc1', '[He]']
    labels = [0.25, 1.5, float('nan'), float('nan')]
    pandas.DataFrame({'smiles': smiles, 'homolumogap': labels}).to_csv(
        folder / 'raw/data.csv.gz', index=False)
    splits = {'train': np.array([1]), 'valid': np.array([0]),
              'test-dev': np.array([2]), 'test-challenge': np.array([3])}
    torch.save(splits, folder / 'split_dict.pt')

    for _ in range(2):  # Initial preprocessing followed by cached reload.
        dataset = PygPCQM4Mv2Dataset(root=str(tmp_path))
        assert len(dataset) == len(smiles)
        for i, text in enumerate(smiles):
            data = dataset[i]
            graph = smiles2graph_pcqm4mv2(text)
            np.testing.assert_array_equal(data.x.numpy(), graph['node_feat'])
            np.testing.assert_array_equal(data.edge_index.numpy(),
                                          graph['edge_index'])
            np.testing.assert_array_equal(data.edge_attr.numpy(),
                                          graph['edge_feat'])
            np.testing.assert_equal(data.y.numpy(), [labels[i]])
        for split, indices in dataset.get_idx_split().items():
            assert torch.equal(indices, torch.from_numpy(splits[split]))

    (folder / 'processed/geometric_data_processed.pt').unlink()
    seen = []

    def custom(text):
        seen.append(text)
        return smiles2graph('CC')

    dataset = PygPCQM4Mv2Dataset(root=str(tmp_path), smiles2graph=custom)
    assert seen == smiles
    assert len(dataset) == len(smiles)
    assert all(data.num_nodes == 2 for data in dataset)


def test_dataset_load_supports_torch_without_weights_only(monkeypatch):
    torch = pytest.importorskip('torch')
    from ogb.lsc.pcqm4mv2 import _load_dataset_file

    def legacy_load(path):
        assert path == 'split_dict.pt'
        return {'train': np.array([1])}

    monkeypatch.setattr(torch, 'load', legacy_load)
    np.testing.assert_array_equal(_load_dataset_file('split_dict.pt')['train'],
                                  [1])
