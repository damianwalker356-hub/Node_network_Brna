# Data, weights, and large files

No external dataset or model weights are required for v0. Synthetic observations are generated in `src/brna/proto.py` using NumPy's seeded generator. The historical JSON run record was supplied with the repository; its original generation provenance is limited to fields present in that record. No external source or dataset license is applicable to the synthetic generator.
