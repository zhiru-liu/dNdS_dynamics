#!/usr/bin/env python3
"""Download NCBI cultured-isolate genomes for one species to the external drive.

Default target: Alistipes putredinis -> /Volumes/Botein/ncbi_isolates/Alistipes_putredinis/

Layout produced per species:
    <out_root>/<Species_name>/
        raw_reports.json                NCBI Datasets v2 paginated assembly reports (full)
        manifest_all_isolates.tsv       one row per cultured isolate (after MAG filter & GCA/GCF dedup)
        manifest_to_download.tsv        subset actually requested (after --hq / --complete / --max)
        fasta/<ACCESSION>.fna.gz        gzipped genome FASTA
        metadata/<ACCESSION>_assembly_data_report.jsonl   per-genome NCBI metadata blob
        download_log.tsv                append-only status log

Examples:
    # All cultured isolates of A. putredinis
    python analysis/download_ncbi_isolates.py

    # Only the high-quality tier (complete + vhq + hq)
    python analysis/download_ncbi_isolates.py --hq

    # A different species, only closed genomes
    python analysis/download_ncbi_isolates.py --species "Bacteroides fragilis" --complete-only

    # See what would be downloaded without fetching
    python analysis/download_ncbi_isolates.py --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from dnds_dynamics import config  # noqa: E402

DATASETS_CLI = str(config.DATASETS_BIN)
DEFAULT_OUT = config.NCBI_ISOLATES_ROOT
DEFAULT_SPECIES = "Alistipes putredinis"
DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"


# --------------------------------------------------------------------------- #
# NCBI Datasets API: fetch all assembly reports for a taxon, with retry.      #
# --------------------------------------------------------------------------- #
def fetch_assembly_reports(species: str) -> list[dict]:
    reports: list[dict] = []
    token = ""
    while True:
        url = (f"{DATASETS_API}/genome/taxon/"
               f"{urllib.parse.quote(species)}/dataset_report?page_size=1000")
        if token:
            url += f"&page_token={token}"
        data = _http_json(url)
        reports.extend(data.get("reports", []))
        token = data.get("next_page_token") or ""
        if not token:
            return reports


def _http_json(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 + 2 * attempt)
            print(f"  retry after {exc!r}", file=sys.stderr)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Isolate vs MAG classifier (mirrors the batch-survey script).                #
# --------------------------------------------------------------------------- #
def is_mag(r: dict) -> bool:
    info = r.get("assembly_info") or {}
    if info.get("assembly_type") == "derived from metagenome":
        return True
    org_name = ((r.get("organism") or {}).get("organism_name") or "").lower()
    bs = info.get("biosample") or {}
    attrs = {a.get("name", "").lower(): a.get("value", "")
             for a in bs.get("attributes", [])}
    bp_title = ""
    bpl = info.get("bioproject_lineage") or []
    if bpl and bpl[0].get("bioprojects"):
        bp_title = (bpl[0]["bioprojects"][0].get("title") or "").lower()
    candidates = [attrs.get(k, "") for k in (
        "isolate", "isolation_source", "env_package", "sample_type",
        "derived_from", "assembly_method", "metagenome_source",
    )] + [org_name, bp_title]
    for v in candidates:
        v = (v or "").lower()
        if "metagenom" in v or "metaspades" in v or "megahit" in v or "bin." in v:
            return True
    return False


def quality_tier(r: dict) -> str:
    info = r.get("assembly_info") or {}
    stats = r.get("assembly_stats") or {}
    ci = r.get("checkm_info") or {}
    c = ci.get("completeness")
    co = ci.get("contamination")
    n50 = int(stats.get("contig_n50") or 0)
    ctgs = int(stats.get("number_of_contigs") or 0)
    if info.get("assembly_level") == "Complete Genome":
        return "complete"
    if c is not None and co is not None:
        if c >= 98 and co <= 2 and (n50 >= 100_000 or (ctgs and ctgs <= 50)):
            return "vhq"
        if c >= 95 and co <= 5:
            return "hq"
        if c >= 90:
            return "mq"
        return "lq"
    return "unk"


def dedup_gca_gcf(reports: list[dict]) -> list[dict]:
    """Same physical assembly is paired GCA/GCF; keep GCF when both present."""
    by_sfx: dict[str, dict] = {}
    for r in reports:
        acc = r.get("accession", "")
        sfx = acc.split("_", 1)[1] if "_" in acc else acc
        prior = by_sfx.get(sfx)
        if prior is None:
            by_sfx[sfx] = r
        elif prior["accession"].startswith("GCA_") and acc.startswith("GCF_"):
            by_sfx[sfx] = r
    return list(by_sfx.values())


def extract_row(r: dict) -> dict:
    info = r.get("assembly_info") or {}
    stats = r.get("assembly_stats") or {}
    ci = r.get("checkm_info") or {}
    org = r.get("organism") or {}
    bs = info.get("biosample") or {}
    attrs = {a.get("name", "").lower(): a.get("value", "")
             for a in bs.get("attributes", [])}
    return {
        "accession": r.get("accession", ""),
        "organism": org.get("organism_name", ""),
        "strain": (org.get("infraspecific_names") or {}).get("strain", "")
                  or attrs.get("strain", ""),
        "isolate": attrs.get("isolate", ""),
        "assembly_level": info.get("assembly_level", ""),
        "assembly_name": info.get("assembly_name", ""),
        "release_date": info.get("release_date", ""),
        "submitter": info.get("submitter", ""),
        "biosample": bs.get("accession", ""),
        "bioproject": info.get("bioproject_accession", ""),
        "isolation_source": attrs.get("isolation_source", ""),
        "host": attrs.get("host", ""),
        "geo_loc": attrs.get("geo_loc_name", ""),
        "collection_date": attrs.get("collection_date", ""),
        "checkm_completeness": ci.get("completeness", ""),
        "checkm_contamination": ci.get("contamination", ""),
        "contig_n50": stats.get("contig_n50", ""),
        "scaffold_n50": stats.get("scaffold_n50", ""),
        "num_contigs": stats.get("number_of_contigs", ""),
        "total_length": stats.get("total_sequence_length", ""),
        "tier": quality_tier(r),
    }


# --------------------------------------------------------------------------- #
# Manifest IO                                                                 #
# --------------------------------------------------------------------------- #
def write_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def append_log(path: Path, accession: str, status: str, detail: str = "") -> None:
    fresh = not path.exists()
    with path.open("a") as fh:
        if fresh:
            fh.write("timestamp\taccession\tstatus\tdetail\n")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fh.write(f"{ts}\t{accession}\t{status}\t{detail}\n")


# --------------------------------------------------------------------------- #
# Download via NCBI `datasets` CLI                                            #
# --------------------------------------------------------------------------- #
def datasets_download(accession: str, zip_path: Path, timeout: int = 120) -> None:
    """Download a single genome's FASTA.

    Per-accession (not --inputfile batches) and ``--include genome`` only: the
    batched / seq-report path stalls indefinitely on large multi-genome zips
    (NCBI-side packaging hang), whereas single-genome ``--include genome``
    transfers complete in a few seconds. A hard timeout guards against hangs.
    """
    cmd = [
        DATASETS_CLI, "download", "genome", "accession", accession,
        "--include", "genome",
        "--filename", str(zip_path),
        "--no-progressbar",
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def unpack_zip(zip_path: Path, fasta_dir: Path, meta_dir: Path,
               log_path: Path) -> int:
    """Extract per-accession FASTA (gzip in place) + metadata jsonl. Return # written."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            parts = member.split("/")
            # NCBI layout: ncbi_dataset/data/<ACCESSION>/<ACCESSION>_<name>_genomic.fna
            if len(parts) >= 4 and parts[1] == "data" and member.endswith("_genomic.fna"):
                accession = parts[2]
                dest = fasta_dir / f"{accession}.fna.gz"
                if dest.exists():
                    continue
                with zf.open(member) as src, gzip.open(dest, "wb",
                                                       compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                append_log(log_path, accession, "downloaded",
                           f"{dest.stat().st_size} bytes")
                n += 1
            elif member.endswith("assembly_data_report.jsonl"):
                # One report for the whole batch — split per-accession
                with zf.open(member) as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        acc = rec.get("accession")
                        if acc:
                            (meta_dir / f"{acc}.json").write_text(
                                json.dumps(rec, indent=2))
    return n


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=DEFAULT_SPECIES,
                    help=f"NCBI species name (default: {DEFAULT_SPECIES!r})")
    ap.add_argument("--out", default=str(DEFAULT_OUT), type=Path,
                    help="Output root (default: %(default)s)")
    ap.add_argument("--hq", action="store_true",
                    help="Only download tiers complete/vhq/hq (CheckM ≥ 95%%, cont ≤ 5%%)")
    ap.add_argument("--complete-only", action="store_true",
                    help="Only single-contig closed genomes")
    ap.add_argument("--max", type=int, default=0,
                    help="Stop after N genomes (debug; 0 = no limit)")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="(deprecated; downloads are now per-accession)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-genome datasets-CLI timeout in seconds (default: 120)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write the two manifests but skip downloading")
    args = ap.parse_args()

    if not Path(DATASETS_CLI).exists():
        sys.exit(f"datasets CLI not found at {DATASETS_CLI}")

    out_root = args.out
    if not out_root.parent.exists():
        sys.exit(f"Output root parent missing: {out_root.parent} "
                 "(is the external drive mounted?)")

    species_slug = args.species.replace(" ", "_")
    out = out_root / species_slug
    fasta_dir = out / "fasta"
    meta_dir = out / "metadata"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "download_log.tsv"

    print(f"Species : {args.species}")
    print(f"Target  : {out}")

    # 1. Inventory
    print("Fetching assembly reports from NCBI Datasets API...")
    raw = fetch_assembly_reports(args.species)
    print(f"  {len(raw)} raw reports")
    (out / "raw_reports.json").write_text(json.dumps(raw, indent=2))

    uniq = dedup_gca_gcf(raw)
    print(f"  {len(uniq)} unique physical assemblies (after GCA/GCF dedup)")

    iso = [r for r in uniq if not is_mag(r)]
    print(f"  {len(iso)} cultured isolates ({len(uniq) - len(iso)} MAGs excluded)")

    rows = [extract_row(r) for r in iso]
    write_manifest(rows, out / "manifest_all_isolates.tsv")

    # 2. Filter
    if args.complete_only:
        rows = [r for r in rows if r["tier"] == "complete"]
        print(f"  --complete-only: {len(rows)} kept")
    elif args.hq:
        rows = [r for r in rows if r["tier"] in ("complete", "vhq", "hq")]
        print(f"  --hq: {len(rows)} kept")

    if args.max and len(rows) > args.max:
        rows = rows[: args.max]
        print(f"  --max: truncated to {len(rows)}")

    write_manifest(rows, out / "manifest_to_download.tsv")
    print(f"  manifest_to_download.tsv: {len(rows)} accessions queued")

    if args.dry_run:
        print("Dry run; not downloading.")
        return 0

    # 3. Download (resumable: skip accessions whose FASTA already exists)
    have = {p.name.split(".fna.gz")[0] for p in fasta_dir.glob("*.fna.gz")}
    to_get = [r["accession"] for r in rows if r["accession"] not in have]
    print(f"  on disk: {len(have)};  to fetch: {len(to_get)}")

    if not to_get:
        print("Nothing to download.")
        return 0

    with tempfile.TemporaryDirectory(prefix="ncbi_iso_") as work:
        work_dir = Path(work)
        written_total = 0
        for k, acc in enumerate(to_get, 1):
            zip_path = work_dir / f"{acc}.zip"
            ok = False
            for attempt in (1, 2):  # one retry on timeout/transient failure
                try:
                    datasets_download(acc, zip_path, timeout=args.timeout)
                    written = unpack_zip(zip_path, fasta_dir, meta_dir, log_path)
                    written_total += written
                    ok = True
                    break
                except subprocess.TimeoutExpired:
                    print(f"    {acc}: timeout (attempt {attempt}/2)")
                except subprocess.CalledProcessError as exc:
                    print(f"    {acc}: datasets exit {exc.returncode} (attempt {attempt}/2)")
                finally:
                    zip_path.unlink(missing_ok=True)
            if not ok:
                append_log(log_path, acc, "error", "download failed after 2 attempts")
            if k % 10 == 0 or k == len(to_get):
                print(f"  {k}/{len(to_get)} processed; {written_total} FASTAs written "
                      f"at {datetime.now(timezone.utc).strftime('%H:%M:%S')}")

    final = sorted(fasta_dir.glob("*.fna.gz"))
    print(f"\nDone. {len(final)} FASTA files in {fasta_dir}")
    print(f"  manifest: {out / 'manifest_to_download.tsv'}")
    print(f"  log:      {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
