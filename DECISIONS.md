# Decision log

| Date | Decision | Options considered | Criteria and rationale | Decider | Status |
|---|---|---|---|---|---|
| 2026-11-26 | Preserve historical run record as immutable source input and copy into versioned results path | Replace / copy unchanged / recompute | Preserve provenance; no invented or silently recomputed evidence | Project owner decision pending | Draft |
| 2026-11-26 | Use NumPy-only toy v0 and keep open-weight adapter as stub | Add framework dependency / keep optional / defer backbone | Small software-only reproducible baseline; no source/model selection evidence | Project owner decision pending | Draft |
| 2026-11-26 | Honor the user's from-scratch constraint with a bounded cooperative character-RNN experiment; make no frontier-parity claim | Permit pretrained weights / implement small from-scratch experiment / promise frontier parity | User selected from-scratch training and accepted less than frontier capability; observed runtime has limited CPU/memory and no detected GPU | User direction | Draft |
