"""Download a prefix from an S3-compatible object store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3
import typer
from dotenv import load_dotenv


def _parse_source(source: str) -> tuple[str, str]:
    if not source.startswith("s3://"):
        raise typer.BadParameter("Source must use the s3://bucket/prefix format.")

    bucket, separator, prefix = source[5:].partition("/")
    if not bucket or not separator:
        raise typer.BadParameter("Source must include a bucket and folder prefix.")
    return bucket, prefix.rstrip("/") + "/"


def _local_path(destination: Path, key: str, prefix: str) -> Path:
    relative_key = key.removeprefix(prefix)
    relative_parts = [part for part in relative_key.split("/") if part not in ("", ".")]
    if any(part == ".." for part in relative_parts):
        raise ValueError(f"Refusing to write outside destination: {key}")
    return destination.joinpath(*relative_parts)


def download_folder(
    source: str,
    destination: Path,
    *,
    endpoint_url: str | None = None,
    region: str | None = None,
    profile: str | None = None,
    no_sign_request: bool = False,
) -> int:
    """Download all objects below an S3 prefix and return the object count."""
    bucket, prefix = _parse_source(source)
    session = boto3.Session(profile_name=profile, region_name=region)
    client_args: dict[str, Any] = {"endpoint_url": endpoint_url}
    if no_sign_request:
        client_args["config"] = boto3.session.Config(signature_version="unsigned")
    client = session.client("s3", **client_args)

    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            target = _local_path(destination, key, prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            typer.echo(f"Downloaded {key} -> {target}")
            count += 1
    return count


app = typer.Typer(help="Download a folder from an S3-compatible object store.")


@app.command()
def download(
    source: str = typer.Argument(
        help="Object-store folder, for example s3://bucket/path/"
    ),
    destination: Path = typer.Argument(help="Local directory to populate."),  # noqa: B008
    endpoint_url: str | None = typer.Option(
        None,
        "--endpoint-url",
        envvar="AWS_ENDPOINT_URL_S3",
        help="S3-compatible API endpoint, such as a MinIO or Ceph endpoint.",
    ),
    region: str | None = typer.Option(None, "--region", envvar="AWS_DEFAULT_REGION"),
    profile: str | None = typer.Option(None, "--profile", envvar="AWS_PROFILE"),
    no_sign_request: bool = typer.Option(
        False,
        "--no-sign-request",
        help="Make unsigned requests for a publicly readable bucket.",
    ),
) -> None:
    """Download every object below SOURCE into DESTINATION."""
    count = download_folder(
        source,
        destination,
        endpoint_url=endpoint_url,
        region=region,
        profile=profile,
        no_sign_request=no_sign_request,
    )
    typer.echo(f"Downloaded {count} object(s).")


def main() -> None:
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
