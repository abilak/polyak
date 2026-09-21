# Dataset manifest and provenance

The executable registry is `configs/datasets.yaml`.  Raw and processed data are ignored by
version control; the repository stores acquisition and conversion code, not republished
third-party datasets.

## SynergyFinder examples

`sparse-ecp fetch-examples` downloads the current Bioconductor package archive and extracts
the documented `NCATS_screening_data`, `ONEIL_screening_data`, and
`mathews_screening_data` objects.  The downloaded NCATS table contains exactly 2,400 rows:
two representative triple-combination blocks with one released response per point on
10 x 10 x 12 dose grids.  The SynergyFinder tutorial describes the underlying assay as
having four replicates, but the released object has no replicate identifier or replicate-level
rows.  It therefore cannot test replicate noise.  It is suitable for an end-to-end
illustration, not for the chat's proposed full 16-combination or noisy-measurement validation
claims.

## NCATS malaria pairwise screen

The 2015 study reports 13,910 combination screens and 728,216 measurements, including a
complete 56-drug/1,540-pair stage on three *P. falciparum* strains.  Its data were hosted by
the legacy NCATS Matrix service.  As of this implementation, the cited `tripod.nih.gov`
endpoint redirects to the general NCATS site, so the code does not pretend the full matrix
was downloaded.  A saved Matrix export can be ingested with the generic SynergyFinder-style
preparer after its columns are mapped to `drug1`, `drug2`, `conc1`, `conc2`, `response`, and
the strain context.

The paper-linked MIT-licensed Synergy Maps repository does retain a processed table of
1,162 QC-retained pair-level synergy records spanning 56 compounds.  It is registered as
`ncats_malaria_pair_scores`, but it contains summary metrics rather than the 6 x 6 response
surfaces and therefore cannot replace the desired within-support dose experiment.

## NCI-ALMANAC

The original dose-level archive contains approximately 2.8 million dose combinations.
The former NCI wiki URL currently returns HTTP 403.  A public Argonne/CANDLE mirror exposes
`ComboDrugGrowth_Nov2017.csv` (about 583 MB), and CellMiner exposes the smaller pair-level
ComboScore table.  The experiment needs the dose-level file; the pair-level file cannot
test continuous optimization within support.

## O'Neil

The Bioconductor package supplies two representative O'Neil blocks.  The full study has a
different distribution route.  The runner accepts a locally supplied full table but does
not label the two-block example as the 583-pair dataset.

## Canonical prepared schema

Prepared CSVs contain:

- `task_id`: cell line, strain, or a robust aggregate;
- `support_id`: canonical sorted drug names joined with `|`;
- `objective`: response to maximize, optionally minus a preregistered dose penalty;
- `response`, `response_std`, and `replicates` when available;
- `x_0 ... x_{d-1}`: zero for absent drugs and per-drug normalized log-dose otherwise;
- original drug, concentration, block, and context columns retained as metadata.

Positive doses map to [0.05, 1] after per-drug log scaling; an actual zero dose stays zero.
This prevents the smallest positive tested concentration from being confused with absence.

## Materials Project Matbench

The official Matbench v0.1 downloads add three independent finite candidate tables:

- `matbench_steels`: 312 experimental alloy yield-strength records;
- `matbench_expt_gap`: 4,604 experimental composition/band-gap records;
- `matbench_perovskites`: 18,928 computed structures and formation energies.

The composition preparer parses formulas and aggregates three exact duplicate compositions in
the band-gap table, leaving 4,601 candidates. Perovskite sites are encoded by crystallographic
role because the table includes mixed-anion structures. See `docs/MATERIALS.md` for the
objectives and limitations.

## CADS oxidative coupling of methane

The CADS table has 12,708 complete reaction rows, 59 catalysts, 13 supports, 26 active-metal
elements, and four encoded process variables. It is the primary catalyst/process benchmark.
The source page prohibits redistribution and requires citation of Nguyen et al., *ACS
Catalysis* 2020, DOI 10.1021/acscatal.9b04293. The repository downloads it directly for local
use and keeps both raw and prepared copies out of version control.
