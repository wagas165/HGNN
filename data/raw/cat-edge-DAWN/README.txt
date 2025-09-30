A hypergraph dataset where a hyperedge corresponds to the set of drugs a patient
reported in a visit to the ER, as reported by the Drug Abuse Warning Network
(DAWN). Hyperedge categories are the most common patient disposition (where the
patient went after the visit) for that drug combination.

The file hyperedges.txt contains lists of drugs that appeared together in a
patient.

Each line lists the drug numbers that appeared together in a patient.

The file hyperedge-labels.txt lists the disposition type label corresponding to
each line (patient) in the hyperedges.txt file.

The file hyperedge-label-identities.txt lists the names of the 10 dispositions,
with order in the list corresponding to the number label used in the file
hyperedge-labels.txt.

The file node-labels.txt lists the identity of all the drugs, with number on the
left corresponding to the number label used in the file hyperedges.txt.

Raw data obtained from https://www.datafiles.samhsa.gov/study-series/drug-abuse-warning-network-dawn-nid13516.
