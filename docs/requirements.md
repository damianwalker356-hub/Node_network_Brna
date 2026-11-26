# Requirements and traceability

**Status: Draft.** Requirement IDs in `traceability.csv` are project-level implementation identifiers, not a claim of NPR compliance. The Rev B T0–T9 cases are specified from the supplied description; the Rev B source itself was not present in the repository tree inspected for this work.

Every test declares its requirement ID using both `@pytest.mark.req('ID')` and its test docstring; the traceability test verifies mapping against `compliance/traceability.csv`.
