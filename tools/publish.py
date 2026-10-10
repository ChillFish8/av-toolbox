#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Publishes every workspace crate to crates.io, waiting out rate limits and skipping crates already published."""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

USER_AGENT = "av-toolbox-publish (https://github.com/ChillFish8/av-toolbox)"
CRATES_API = "https://crates.io/api/v1/crates"
PUBLISH_COMMAND = ["cargo", "publish", "--workspace", "--features", "av-toolbox-gpu/vulkan"]
RETRY_AFTER = re.compile(r"try again after (.+? GMT)")
RETRY_MARGIN_SECONDS = 15
MAX_ATTEMPTS = 20


class PublishError(Exception):
    pass


@dataclass(frozen=True)
class Crate:
    name: str
    version: str


def workspace_crates() -> list[Crate]:
    command = ["cargo", "metadata", "--no-deps", "--format-version", "1"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    metadata = json.loads(output)

    # An empty `publish` list marks a crate that is never published.
    packages = [package for package in metadata["packages"] if package.get("publish") != []]
    return [Crate(package["name"], package["version"]) for package in packages]


def is_published(crate: Crate) -> bool:
    url = f"{CRATES_API}/{crate.name}/{crate.version}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request):
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False

        raise PublishError(f"crates.io answered {error.code} when looking up {crate.name}") from error


def run_publish(published: list[Crate], extra_arguments: list[str]) -> tuple[int, str]:
    """Runs `cargo publish`, echoing its output as it goes and handing it back for inspection."""
    excludes = [argument for crate in published for argument in ("--exclude", crate.name)]
    command = PUBLISH_COMMAND + excludes + extra_arguments
    print(f"$ {' '.join(command)}", flush=True)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        lines.append(line)

    status = process.wait()
    return status, "".join(lines)


def retry_time(output: str) -> datetime | None:
    if "429" not in output:
        return None

    found = RETRY_AFTER.search(output)
    if found is None:
        return None

    return parsedate_to_datetime(found.group(1))


def wait_until(moment: datetime) -> None:
    now = datetime.now(timezone.utc)
    seconds = (moment - now).total_seconds() + RETRY_MARGIN_SECONDS
    seconds = max(seconds, RETRY_MARGIN_SECONDS)

    print(f"rate limited by crates.io, retrying at {moment.isoformat()} ({seconds:.0f}s)", flush=True)
    time.sleep(seconds)


def publish(extra_arguments: list[str]) -> int:
    crates = workspace_crates()
    dry_run = "--dry-run" in extra_arguments

    for _ in range(MAX_ATTEMPTS):
        published = [crate for crate in crates if is_published(crate)]
        if len(published) == len(crates):
            print("every crate is already published")
            return 0

        for crate in published:
            print(f"skipping {crate.name} {crate.version}, already on crates.io")

        status, output = run_publish(published, extra_arguments)
        if status == 0:
            return 0

        moment = retry_time(output)
        if moment is None or dry_run:
            return status

        wait_until(moment)

    raise PublishError(f"still rate limited after {MAX_ATTEMPTS} attempts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Any other arguments are passed on to `cargo publish`.")
    _, cargo_arguments = parser.parse_known_args()

    try:
        return publish(cargo_arguments)
    except PublishError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    status = main()
    sys.exit(status)
