# Isolate genome pipeline (`analysis/`)

Builds the same kind of SNV catalog from **cultured isolate genomes** that the
main study builds from metagenomes, so the two can be compared head-to-head. The
steps, in order:

1. `download_ncbi_isolates.py` — fetch cultured-isolate assemblies from NCBI.
2. `build_isolate_snv_table.py` — align each assembly to a reference (nucmer) and
   aggregate into an SNV table (`--midas-species` selects the species/reference).
3. `run_isolate_cphmm.py` — detect recombination with the CP-HMM.
4. `compute_identical_fraction.py`, `compute_isolate_dnds.py`,
   `compute_stratified_isolate_dnds.py` — clonal fraction and stratified dN/dS.

Most of the reusable logic is in step 1, documented below.

---

## Downloading isolates: `download_ncbi_isolates.py`

### What it does

For one species it queries the **NCBI Datasets v2 API** for *every* assembly,
then narrows that set down to cultured isolates of a chosen quality:

```
raw assembly reports          (NCBI Datasets API, all assemblies for the taxon)
  └─ collapse GCA/GCF pairs    (same physical genome submitted twice; keep RefSeq/GCF)
       └─ drop MAGs            (keep cultured isolates only — see below)
            └─ quality tier    (complete / vhq / hq / mq / lq / unk)
                 └─ download    (optionally restricted to high-quality tiers)
```

Worked example (`--species "Bacteroides fragilis"`, run 2026):
`3376 raw → 2603 unique → 2043 cultured isolates (560 MAGs excluded)`, of which
`963` are high-quality (`complete 130 + vhq 538 + hq 295`).

### How "isolate" status is determined (`is_mag`)

The key filter is separating **cultured isolates** from **metagenome-assembled
genomes (MAGs)**. An assembly is treated as a MAG (and excluded) if **either**:

1. **NCBI's own flag** — `assembly_info.assembly_type == "derived from
   metagenome"`; **or**
2. **a keyword screen** of its metadata — the organism name, the BioProject
   title, and these BioSample attributes:
   `isolate, isolation_source, env_package, sample_type, derived_from,
   assembly_method, metagenome_source`
   — contains any of `metagenom`, `metaspades`, `megahit`, or `bin.`
   (i.e. metagenomic sampling context or a metagenomic assembler/binning tool).

Anything that passes both checks is kept as a cultured isolate. The keyword screen
catches MAGs that NCBI did not formally flag (the `assembly_type` field is often
unset), so both checks are needed.

### Quality tiers (`quality_tier`)

Assigned from CheckM completeness/contamination, assembly level, and contiguity:

| tier | criteria |
|---|---|
| `complete` | assembly level = "Complete Genome" |
| `vhq` | CheckM ≥ 98% complete, ≤ 2% contam, and (contig N50 ≥ 100 kb or ≤ 50 contigs) |
| `hq` | CheckM ≥ 95% complete, ≤ 5% contam |
| `mq` | CheckM ≥ 90% complete |
| `lq` | CheckM present but below mq |
| `unk` | no CheckM info |

`--hq` keeps `complete + vhq + hq` (the set used in the paper); `--complete-only`
keeps just closed genomes.

### Running it

```bash
# See what would be selected, without downloading (writes manifests only):
python analysis/download_ncbi_isolates.py --species "Bacteroides fragilis" --dry-run

# Download the high-quality tier:
python analysis/download_ncbi_isolates.py --species "Bacteroides fragilis" --hq

# Useful flags: --out <dir>  --complete-only  --max N (debug cap)  --timeout S
```

Downloads are **resumable** — re-running skips accessions whose FASTA already
exists, so an interrupted run just continues.

### Outputs (under `<out>/<Species_name>/`)

| file | contents |
|---|---|
| `raw_reports.json` | full NCBI Datasets reports (the raw inventory) |
| `manifest_all_isolates.tsv` | one row per cultured isolate (after dedup + MAG filter), with metadata and `tier` |
| `manifest_to_download.tsv` | the subset actually requested (after `--hq`/`--complete-only`/`--max`) |
| `fasta/<ACCESSION>.fna.gz` | downloaded genome FASTA |
| `metadata/<ACCESSION>.json` | per-genome NCBI metadata |
| `download_log.tsv` | append-only status log |

### Prerequisites

- The NCBI **`datasets` CLI** on the path configured in `dnds_dynamics/config.py`
  (`DATASETS_BIN`).
- An output location: `--out` defaults to `config.NCBI_ISOLATES_ROOT` (an external
  drive); pass `--out <local_dir>` to write elsewhere. Its parent must exist.
- Network access to `api.ncbi.nlm.nih.gov`.

---

## Downstream

Once `manifest_to_download.tsv` and `fasta/` exist, build the catalog and run the
rest of the pipeline against the matching MIDAS reference:

```bash
python analysis/build_isolate_snv_table.py --midas-species Bacteroides_fragilis_54507
python analysis/run_isolate_cphmm.py <species-preset>     # presets in run_isolate_cphmm.py
python analysis/compute_identical_fraction.py  ...
python analysis/compute_isolate_dnds.py  ...
python analysis/compute_stratified_isolate_dnds.py  ...
```

The resulting SNV tables share their coordinate system and site-type annotations
with the metagenome (QP) catalogs, so downstream dN/dS code is identical for both.
