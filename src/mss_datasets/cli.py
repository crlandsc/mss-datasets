"""CLI entry point for mss-datasets."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

# Load .env before Click parses envvar options (e.g. ZENODO_TOKEN)
load_dotenv()

from mss_datasets import __version__
from mss_datasets.inventory import LINK_MODES
from mss_datasets.pipeline import Pipeline, PipelineConfig


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _load_config_file(config_path: str) -> dict:
    """Load YAML config file and return as flat dict."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Flatten nested 'datasets' key
    flat = {}
    datasets = data.pop("datasets", {})
    for k, v in datasets.items():
        flat[k] = v
    flat.update(data)
    return flat


def _print_summary(result: dict) -> None:
    """Print human-readable summary."""
    if result.get("dry_run"):
        click.echo("\nMSS Datasets — Dry Run")
        click.echo("=" * 40)
        click.echo(f"Profile: {result['profile']}")
        click.echo(f"Total tracks: {result['total_tracks']}")
        click.echo(f"Skipped (overlap): {result['skipped_musdb_overlap']}")
        if result.get("excluded_bleed"):
            click.echo(f"Excluded (bleed): {result['excluded_bleed']}")
        click.echo(f"\nBy dataset: {result['by_dataset']}")
        click.echo(f"By split: {result['by_split']}")
        click.echo(f"Stem folders: {', '.join(result['stem_folders'])}")
        return

    if result.get("error"):
        click.echo(f"\nError: {result['error']}", err=True)
        sys.exit(1)

    click.echo("\nMSS Datasets — Complete")
    click.echo("=" * 40)
    click.echo(f"Profile: {result['profile']}")
    click.echo(f"Total tracks: {result['total_tracks']}")
    if result["skipped_musdb_overlap"]:
        click.echo(f"Deduplicated: {result['skipped_musdb_overlap']} tracks (MedleyDB preferred)")
    if result.get("excluded_bleed"):
        click.echo(f"Excluded (bleed): {result['excluded_bleed']} tracks")
    click.echo(f"Errors: {result['errors']} tracks skipped (see errors.json)")
    click.echo("\nOutput stem counts:")
    for stem, count in result["stem_counts"].items():
        click.echo(f"  {stem + '/':12s} {count} files")
    click.echo(f"\nTotal: {result['total_files']:,} WAV files")
    disk_mb = result["disk_usage_bytes"] / (1024 * 1024)
    if disk_mb > 1024:
        click.echo(f"Disk usage: ~{disk_mb / 1024:.1f} GB")
    else:
        click.echo(f"Disk usage: ~{disk_mb:.0f} MB")


def _resolve_output(output: str, file_config: dict) -> str:
    """CLI --output wins; otherwise fall back to the config file, then the default."""
    if output != "./output":
        return output
    return file_config.get("output", "./output")


def _print_inventory(inv: dict) -> None:
    """Print the regrouped inventory summary."""
    from mss_datasets.inventory import VIEW_LABELS
    from mss_datasets.output_tree import VIEWS

    click.echo("\nMSS Datasets — Inventory")
    click.echo("=" * 52)
    click.echo(f"Tree: {inv['root']}")
    click.echo(f"{inv['total_tracks']} tracks, {inv['total_files']} WAV files\n")

    click.echo("On disk:")
    for source, n in inv["on_disk_by_source"].items():
        click.echo(f"  {source + '/':16s} {n:4d} tracks")

    click.echo("\nRegrouped for review:")
    for view in VIEWS:
        v = inv["views"][view]
        comp = " + ".join(f"{n} from {s}" for s, n in v["by_source"].items())
        click.echo(f"  {VIEW_LABELS[view]:24s} {v['track_count']:4d}   {comp}")
    click.echo(f"  {'TOTAL':24s} {inv['total_tracks']:4d}")


def _run_review(output: str, inventory: bool, audit: bool,
                sorted_view: str | None, link_mode: str) -> None:
    """Handle the read-only review modes against an existing output tree."""
    from mss_datasets.audit import render_audit, run_audit
    from mss_datasets.inventory import build_inventory, build_sorted_view, render_report
    from mss_datasets.output_tree import OutputTree

    try:
        tree = OutputTree(output).load()
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not tree.tracks:
        click.echo(f"Error: no aggregated output found in {output}", err=True)
        sys.exit(1)

    failed = False
    if audit:
        result = run_audit(tree)
        click.echo("\n" + render_audit(result))
        failed = not result["passed"]

    if inventory:
        inv = build_inventory(tree)
        _print_inventory(inv)
        report_path = tree.metadata_dir / "inventory.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(inv))
        click.echo(f"\nWrote {report_path}")

    if sorted_view:
        stats = build_sorted_view(tree, sorted_view, link_mode=link_mode)
        click.echo(f"\nBuilt review tree at {stats['dest']} "
                   f"({stats['link_mode']}, {stats['created']} entries)")

    if failed:
        sys.exit(1)


def _print_download_summary(results: dict) -> None:
    """Print download results summary."""
    click.echo("\nMSS Datasets — Download Summary")
    click.echo("=" * 40)
    for name, path in results.items():
        if path:
            click.echo(f"  {name}: {path}")
        else:
            click.echo(f"  {name}: skipped")


@click.command()
@click.version_option(version=__version__, prog_name="mss-datasets")
@click.option("--musdb18hq-path", type=click.Path(exists=True), default=None,
              help="Path to MUSDB18-HQ dataset")
@click.option("--moisesdb-path", type=click.Path(exists=True), default=None,
              help="Path to MoisesDB dataset")
@click.option("--medleydb-path", type=click.Path(exists=True), default=None,
              help="Path to MedleyDB dataset")
@click.option("--output", "-o", type=click.Path(), default="./output",
              help="Output directory")
@click.option("--profile", type=click.Choice(["vdbo", "vdbo+gp"]), default="vdbo",
              help="Stem profile")
@click.option("--workers", type=int, default=1,
              help="Number of parallel workers")
@click.option("--include-mixtures", is_flag=True, default=False,
              help="Generate mixture files")
@click.option("--group-by-dataset", is_flag=True, default=False,
              help="Add source dataset subfolders within each stem folder")
@click.option("--split-output", is_flag=True, default=False,
              help="Organize output into train/ and val/ directories")
@click.option("--include-bleed", is_flag=True, default=False,
              help="Include tracks with stem bleed (excluded by default)")
@click.option("--verify-mixtures", is_flag=True, default=False,
              help="Verify stem sums match original mixtures")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be processed without writing files")
@click.option("--config", "config_file", type=click.Path(exists=True), default=None,
              help="Path to YAML config file")
@click.option("--download", is_flag=True, default=False,
              help="Download datasets")
@click.option("--aggregate", is_flag=True, default=False,
              help="Aggregate datasets into unified stem folders")
@click.option("--inventory", is_flag=True, default=False,
              help="Report an existing output tree regrouped for review (read-only)")
@click.option("--audit", is_flag=True, default=False,
              help="Check an existing output tree for duplicates and split leakage")
@click.option("--sorted-view", type=click.Path(), default=None,
              help="Build the regrouped review tree at this path (implies --inventory)")
@click.option("--link-mode", type=click.Choice(LINK_MODES), default="symlink",
              help="How --sorted-view materializes files (default: symlink, no copies)")
@click.option("--data-dir", type=click.Path(), default="./datasets",
              help="Directory for raw dataset downloads")
@click.option("--zenodo-token", default=None, envvar="ZENODO_TOKEN",
              help="Zenodo access token for MedleyDB (also: ZENODO_TOKEN env var)")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Verbose logging")
def main(
    musdb18hq_path, moisesdb_path, medleydb_path, output, profile,
    workers, include_mixtures, group_by_dataset, split_output,
    include_bleed, verify_mixtures, dry_run, config_file,
    download, aggregate, inventory, audit, sorted_view, link_mode,
    data_dir, zenodo_token, verbose,
):
    """Aggregate multiple MSS datasets into unified stem folders."""
    _setup_logging(verbose)

    # Review modes read an existing output tree — they never re-run aggregation,
    # so --config here only supplies the output path.
    run_review = inventory or audit or (sorted_view is not None)

    # Infer aggregate mode from --config or --dry-run
    run_aggregate = aggregate or dry_run or (config_file is not None and not run_review)

    if not download and not run_aggregate and not run_review:
        click.echo(
            "Error: Specify at least one mode: --download, --aggregate, --dry-run, "
            "--inventory, or --audit",
            err=True,
        )
        sys.exit(1)

    # Load config file early so download options (data_dir, zenodo_token) are available
    file_config = {}
    if config_file:
        file_config = _load_config_file(config_file)

    # Apply config defaults for download-related options (CLI flags override)
    if data_dir == "./datasets" and file_config.get("data_dir"):
        data_dir = file_config["data_dir"]
    if zenodo_token is None and file_config.get("zenodo_token"):
        zenodo_token = file_config["zenodo_token"]

    if run_review:
        _run_review(
            _resolve_output(output, file_config),
            inventory=inventory or sorted_view is not None,
            audit=audit,
            sorted_view=sorted_view,
            link_mode=link_mode,
        )
        return

    # Handle download mode
    if download:
        from mss_datasets.download import download_all

        results = download_all(Path(data_dir), zenodo_token)
        if results["musdb18hq"] and not musdb18hq_path:
            musdb18hq_path = str(results["musdb18hq"])
        if results["medleydb"] and not medleydb_path:
            medleydb_path = str(results["medleydb"])
        if not run_aggregate:
            _print_download_summary(results)
            return

    # Build pipeline config — CLI values take precedence over config file
    pipeline_config = PipelineConfig(
        musdb18hq_path=musdb18hq_path or file_config.get("musdb18hq_path"),
        moisesdb_path=moisesdb_path or file_config.get("moisesdb_path"),
        medleydb_path=medleydb_path or file_config.get("medleydb_path"),
        output=_resolve_output(output, file_config),
        profile=profile if profile != "vdbo" else file_config.get("profile", "vdbo"),
        workers=workers if workers != 1 else file_config.get("workers", 1),
        include_mixtures=include_mixtures or file_config.get("include_mixtures", False),
        group_by_dataset=group_by_dataset or file_config.get("group_by_dataset", False),
        split_output=split_output or file_config.get("split_output", False),
        include_bleed=include_bleed or file_config.get("include_bleed", False),
        verify_mixtures=verify_mixtures or file_config.get("verify_mixtures", False),
        dry_run=dry_run,
        verbose=verbose,
    )

    # Check that at least one dataset path is provided
    if not any([pipeline_config.musdb18hq_path, pipeline_config.moisesdb_path,
                pipeline_config.medleydb_path]):
        click.echo("Error: At least one dataset path must be provided.", err=True)
        click.echo("Use --musdb18hq-path, --moisesdb-path, or --medleydb-path, or --download", err=True)
        sys.exit(1)

    pipeline = Pipeline(pipeline_config)
    result = pipeline.run()
    _print_summary(result)


if __name__ == "__main__":
    main()
