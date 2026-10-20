"""Preserve PCQM4Mv2's historical molecular graphs with current RDKit.

RDKit 2024.09 removed some previously permitted valences. The published
benchmark contains 39 affected SMILES, reported in OGB issues #498 and #510.
Dropping these rows would invalidate the dataset splits; changing their bonds
or charges would change the benchmark. This converter preserves the original
RDKit 2024.03.6 features for those exact records, not a general chemical
interpretation of other molecules with nonstandard valences.
"""

from rdkit import Chem

from ogb.utils.mol import ReorderCanonicalRankAtoms, _mol2graph, smiles2graph


# Exact strings from the official PCQM4Mv2 release (archive MD5:
# 65b742bafca5670be4497499db7d361b). Do not canonicalize or broaden this set:
# ordinary SMILES, including invalid structures, retain RDKit's strict parser.
_LEGACY_SMILES = frozenset({
    'O[Si]123O[Si]3(O1)(O2)O',
    'C1[SiH2]2[SiH2]1C2',
    'Cl[SiH]12C[SiH2]2C1',
    'Cl[SiH]12C[SiH]2(C1)Cl',
    'Cl[Si]12(Cl)C[SiH]2(C1)Cl',
    'c1ccc(cc1)[SiH]12C[SiH]2(C1)c1ccccc1',
    'Br[SiH]12C[SiH2]2C1',
    'C[Si]1234C[Si]564C42[Si]23(C1)[Si]54(C2)(C6)C',
    'CO[Si]12(C)O[Si]2(O1)(C)OC',
    'CC1(C)[Si]2[Si]1(C2)(C)C',
    'CCO[Si]12(C)C[Si]2(C1)(C)OCC',
    'CC1[Si]2([Si]1(C2)(Cl)Cl)(Cl)Cl',
    'c1ccc(cc1)[Si]12(C[Si]2C1)c1ccccc1',
    'CO[Si]1(OC)(OC)N(C)[C](N1C)N(C)C',
    'C[Si]123O[Si]3(O1)(O2)C',
    'CN1[C](O[P]1(F)(F)(F)F)c1ccccc1',
    '[O]N1O[Si]2(O1)([O])ON(O2)[O]',
    'F[Si](F)(F)(F)(F)F',
    'Cl[C]1C=CC=C2N1C[Si](O2)(C)(C)Cl',
    'C[C]1C=CC=C2N1C[Si](O2)(C)(C)Cl',
    'CO[Si]12(C)C[Si]2(C1)(OC)OC',
    'CO[Si]12(OC)C[Si]2(C1)(OC)OC',
    'F[P](F)(F)(F)(F)F',
    'F[Si]12(F)C[Si]2(C1)(F)F',
    'C[Si]1(C)(Cl)OC2=CC=C[CH]N2C1',
    'C[Si]12(C[Si]2(C1)(C)C)c1ccccc1',
    'C[Si]12(C)C[Si]2C1',
    'C[Si]12(C)C[Si]2(C1)(Cl)Cl',
    'C[Si]12(Cl)C[Si]2(C1)(C)Cl',
    'Fc1ccc(cc1)[C]1OC[Si](O1)(F)(F)F',
    'Cc1ccc(cc1)[C]1OC[Si](O1)(F)(F)F',
    'C=C[Si]12(C)C[Si]2(C1)(C)C',
    'ClC1=CC=C2N([CH]1)C[Si](O2)(C)(C)Cl',
    'C[Si]12(Cl)C[Si]2(C1)(C)C',
    'O[Si]1(O)O[Si]23([Si](O1)(O2)(O3)O)O',
    'Cl[Si]12(Cl)C[Si]2(C1)(Cl)Cl',
    'C=C[Si]12(C)C[Si]2(C1)(C)C=C',
    'CO[Si]12(OC)O[Si]2(O1)(OC)OC',
    '[O]P123OP45([Si]3(O1)(O2)(O4)O5)[O]',
})


def smiles2graph_pcqm4mv2(smiles_string, removeHs=True, reorder_atoms=False):
    """Convert a PCQM4Mv2 record without changing its historical graph."""
    if smiles_string not in _LEGACY_SMILES:
        return smiles2graph(smiles_string, removeHs, reorder_atoms)

    mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
    if mol is None:
        raise ValueError('Cannot parse a legacy PCQM4Mv2 SMILES')

    mol.UpdatePropertyCache(strict=False)
    legacy_radicals = []
    for atom in mol.GetAtoms():
        valence = (sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                   + atom.GetNumExplicitHs())
        limit = {14: 6, 15: 7}.get(atom.GetAtomicNum())
        if (limit is not None and atom.GetFormalCharge() == 0
                and not atom.GetIsAromatic() and valence in (limit - 1, limit)):
            legacy_radicals.append((atom, int(limit - valence)))
        else:
            # Keep valence validation for every atom outside the two historical
            # neutral Si/P states, even within an allowlisted molecule.
            atom.UpdatePropertyCache(strict=True)

    # Only the property step has already been handled above. Retain ring,
    # aromaticity, conjugation, hybridization and all other sanitization.
    Chem.SanitizeMol(
        mol, Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    # Current RDKit assigns zero radicals to these now-unsupported states.
    # Restore the historical values before adding Hs or reordering atoms.
    for atom, radicals in legacy_radicals:
        atom.SetNumRadicalElectrons(radicals)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    if not removeHs:
        mol = Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)
    return _mol2graph(mol)
