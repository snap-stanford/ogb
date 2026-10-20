`pcqm4mv2_legacy_graphs.json` records graph hashes produced by the unmodified
`ogb.utils.smiles2graph` at commit
`61e9784ca76edeaa6e259ba0f836099608ff0586`, using RDKit 2024.03.6 and
NumPy 1.26.4. Its 39 SMILES and zero-based indices were checked against the
official PCQM4Mv2 archive with MD5 `65b742bafca5670be4497499db7d361b`.
The same records are reported in
[OGB issue #510](https://github.com/snap-stanford/ogb/issues/510).

For each record, the reference converter ran all four combinations of
`removeHs=True/False` and `reorder_atoms=True/False`. Each graph's NumPy arrays
were converted with `tolist()`, and the full graph dictionary was serialized
using `json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()`.
The fixture stores the SHA-256 of those bytes, including every node feature,
edge feature, node index and edge ordering. The compatibility implementation
was not used to generate these expected results.

RDKit's 2024.09 valence changes also changed radical assignment for the affected
Si and P atoms. Merely suppressing valence errors produces different graph
hashes; the fixture therefore checks benchmark compatibility, not just parsing.
