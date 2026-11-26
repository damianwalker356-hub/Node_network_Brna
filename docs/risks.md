# Risk register

**Status: Draft.** Risks are preliminary project observations, not a formally approved register.

| ID | Risk | Mitigation / disposition | Status |
|---|---|---|---|
| RSK-01 | Toy synthetic result may not generalize | Clearly bound claims; later select representative tasks | Draft |
| RSK-02 | Unequal parameter counts confound comparison | Implement equal-parameter option and report counts | Draft |
| RSK-03 | Few seeds and untuned training limit conclusions | Default new comparisons to at least 10 seeds; record tuning | Draft |
| RSK-04 | Missing source artifacts prevent exact Rev B/test port and standards crosswalk | Request source artifacts; retain gaps in handoff | Draft |
| RSK-05 | Proposed software class may be incorrect | Verify against NPR 7150.2D Appendix D with responsible authority | Draft |
| RSK-06 | Character-level RNN and small corpus do not establish useful general language capability | Bound product claims; define a separately resourced corpus and benchmark before capability claims | Draft |
| RSK-07 | Repeated agent outputs may reinforce errors or reduce other response quality | Bound reward updates, preserve checkpoints, record feedback runs, and use held-out task checks | Draft |
