import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import build

REVERSING_CONVERTER = """\
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
import sys
from pathlib import Path

sources = Path(sys.argv[1])
output = Path(sys.argv[2])
raw = (sources / "raw.dat").read_bytes()
(output / "out.bin").write_bytes(raw[::-1])
"""


def quoted_list(names) -> str:
    quoted = [f'"{name}"' for name in names]
    return ", ".join(quoted)


def add_dependency(
    repository: Path,
    identifier: str,
    content: bytes = b"weights",
    sha256: str | None = None,
    files=("out.bin",),
    builtin=("out.bin",),
    licence: bool = True,
) -> Path:
    upstream = repository.parent / "upstream" / identifier / "raw.dat"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_bytes(content)

    digest = sha256 or hashlib.sha256(content).hexdigest()
    directory = repository / "dependencies" / identifier
    directory.mkdir(parents=True)

    manifest = (
        f'identifier = "{identifier}"\n'
        f"files = [{quoted_list(files)}]\n"
        f"builtin = [{quoted_list(builtin)}]\n"
        "\n"
        "[[sources]]\n"
        'name = "raw.dat"\n'
        f'url = "{upstream.as_uri()}"\n'
        f'sha256 = "{digest}"\n'
    )
    (directory / "dependency.toml").write_text(manifest)
    (directory / "convert.py").write_text(REVERSING_CONVERTER)

    if licence:
        (directory / "LICENSE").write_text("licence text\n")

    return upstream


def lock_path(repository: Path) -> Path:
    return repository / "av-toolbox-models" / "dependencies.lock"


def reversed_hash(content: bytes = b"weights") -> str:
    reversed_content = content[::-1]
    return hashlib.sha256(reversed_content).hexdigest()


def test_a_dependency_builds_its_zip_lock_and_builtin(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")

    status = build.build([], repository, check=False)

    assert status == 0
    zip_path = repository / "target" / "dependencies" / "optional-dependency-fake.zip"
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["LICENSE", "out.bin"]
        assert archive.read("out.bin") == b"sthgiew"
        dates = [info.date_time for info in archive.infolist()]
        assert dates == [build.ZIP_TIMESTAMP, build.ZIP_TIMESTAMP]

    expected_lock = build.LOCK_HEADER + f'\n[fake]\n"out.bin" = "{reversed_hash()}"\n'
    assert lock_path(repository).read_text() == expected_lock

    builtin = repository / "av-toolbox-models" / "builtin" / "out.bin"
    assert builtin.read_bytes() == b"sthgiew"


def test_a_source_with_the_wrong_sha256_is_refused_and_not_cached(tmp_path):
    repository = tmp_path / "repository"
    wrong = "0" * 64
    add_dependency(repository, "fake", sha256=wrong)

    with pytest.raises(build.BuildError) as refusal:
        build.build([], repository, check=False)

    message = str(refusal.value)
    assert "raw.dat" in message
    assert wrong in message
    cache = repository / "target" / "dependencies" / "cache" / "fake"
    assert list(cache.iterdir()) == []


def test_a_cached_source_is_not_downloaded_again(tmp_path):
    repository = tmp_path / "repository"
    upstream = add_dependency(repository, "fake")
    build.build([], repository, check=False)
    upstream.unlink()

    status = build.build([], repository, check=False)

    assert status == 0


def test_check_passes_after_a_build_and_fails_when_the_lock_differs(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    build.build([], repository, check=False)

    unchanged = build.build([], repository, check=True)
    edited_lock = lock_path(repository).read_text() + "# edited\n"
    lock_path(repository).write_text(edited_lock)
    changed = build.build([], repository, check=True)

    assert unchanged == 0
    assert changed == 1
    assert lock_path(repository).read_text() == edited_lock


def test_check_fails_when_a_builtin_differs(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    build.build([], repository, check=False)
    builtin = repository / "av-toolbox-models" / "builtin" / "out.bin"
    builtin.write_bytes(b"stale")

    status = build.build([], repository, check=True)

    assert status == 1
    assert builtin.read_bytes() == b"stale"


def test_building_one_identifier_keeps_the_others_in_the_lock(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    add_dependency(repository, "other", content=b"other", builtin=())
    kept_section = '\n[zzz]\n"kept.bin" = "' + "a" * 64 + '"\n'
    lock_path(repository).parent.mkdir(parents=True, exist_ok=True)
    lock_path(repository).write_text(build.LOCK_HEADER + kept_section)

    build.build(["fake"], repository, check=False)

    lock = lock_path(repository).read_text()
    assert f'[fake]\n"out.bin" = "{reversed_hash()}"' in lock
    assert "[zzz]" in lock
    assert "[other]" not in lock
    assert lock.index("[fake]") < lock.index("[zzz]")


def test_an_unknown_identifier_is_refused(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")

    with pytest.raises(build.BuildError, match="missing"):
        build.build(["missing"], repository, check=False)


def test_a_converter_writing_other_files_is_refused(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake", files=("different.bin",), builtin=())

    with pytest.raises(build.BuildError, match="convert.py wrote"):
        build.build([], repository, check=False)


def test_a_builtin_not_listed_in_files_is_refused(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake", builtin=("elsewhere.bin",))

    with pytest.raises(build.BuildError, match="elsewhere.bin"):
        build.build([], repository, check=False)


def test_a_missing_licence_is_refused(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake", licence=False)

    with pytest.raises(build.BuildError, match="LICENSE"):
        build.build([], repository, check=False)


def test_zips_are_identical_across_builds(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    zip_path = repository / "target" / "dependencies" / "optional-dependency-fake.zip"

    build.build([], repository, check=False)
    first = zip_path.read_bytes()
    build.build([], repository, check=False)
    second = zip_path.read_bytes()

    assert first == second


def test_two_dependencies_with_the_same_builtin_name_are_refused(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    add_dependency(repository, "other", content=b"other")

    with pytest.raises(build.BuildError, match="out.bin"):
        build.build([], repository, check=False)


def test_a_full_build_drops_lock_sections_of_removed_dependencies(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    build.build([], repository, check=False)
    stale_section = '\n[zzz]\n"gone.bin" = "' + "a" * 64 + '"\n'
    stale_lock = lock_path(repository).read_text() + stale_section
    lock_path(repository).write_text(stale_lock)

    checked = build.build([], repository, check=True)
    rebuilt = build.build([], repository, check=False)

    lock = lock_path(repository).read_text()
    assert checked == 1
    assert rebuilt == 0
    assert "[zzz]" not in lock
    assert "[fake]" in lock


def test_a_failing_converter_is_reported_naming_the_dependency(tmp_path):
    repository = tmp_path / "repository"
    add_dependency(repository, "fake")
    failing = REVERSING_CONVERTER + "sys.exit(1)\n"
    converter = repository / "dependencies" / "fake" / "convert.py"
    converter.write_text(failing)

    with pytest.raises(build.BuildError, match="fake"):
        build.build([], repository, check=False)
