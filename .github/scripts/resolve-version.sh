#!/usr/bin/env bash
# Resolves the version a release workflow builds and checks it against the workspace version.
#
# The version is the release tag on a release, or the dispatch input, or the workspace version when the input is
# empty. A mismatch fails a release and only warns on a dispatch, which never publishes anything.
set -euo pipefail

cargo_version="$(python3 -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('Cargo.toml').read_text())['workspace']['package']['version'])")"

if [ "${EVENT_NAME}" = "release" ]; then
    version="${RELEASE_TAG}"
else
    version="${DISPATCH_VERSION:-${cargo_version}}"
fi

version="${version#v}"

if ! [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "::error::'${version}' is not a valid semantic version (expected X.Y.Z or X.Y.Z-suffix)"
    exit 1
fi

if [ "${cargo_version}" != "${version}" ]; then
    message="version ${version} does not match Cargo.toml ${cargo_version}"

    if [ "${EVENT_NAME}" = "release" ]; then
        echo "::error::${message}"
        exit 1
    fi

    echo "::warning::${message}"
fi

echo "version=${version}" >> "${GITHUB_OUTPUT}"
