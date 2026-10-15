#!/usr/bin/env python3
"""Build a publishable snapshot of this repository.

Reads tracked files only. The source tree is never rewritten. LICENSE and
README.md must already be tracked. Credential filenames, secret values,
local paths, account names, email addresses, and intranet hosts fail the
check. An existing snapshot is left in place when the check fails.
"""

from __future__ import annotations

import getpass
import io
import ipaddress
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.name
DEST = ROOT / f"{PROJECT}-pub"
TMP = ROOT / f"{PROJECT}-pub.tmp"
BACKUP = ROOT / f"{PROJECT}-pub.replacing"
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAMES = {SCRIPT_PATH.name, "_public.py", "public_.py", "public_", "public.py"}

REQUIRED_FILES = ("LICENSE", "README.md")
ENV_PATHS = (".env", "nested/.env")
ENV_EXAMPLE_NAMES = {".env.example", ".env.sample", ".env.template"}
CREDENTIAL_NAMES = {
    ".env",
    ".envrc",
    ".pypirc",
    "credentials.json",
    "credentials.yml",
    "credentials.yaml",
    "secrets.json",
    "secrets.yml",
    "secrets.yaml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
CREDENTIAL_SUFFIXES = {".pem", ".p12", ".pfx", ".key"}
# Publish policy for this project. These are not secret-detection rules.
# data/ is part of the published tree. private/ is not.
REJECT_CHINESE = True
DATA_DIRECTORIES = {"private"}
DATA_SUFFIXES = {".pt", ".pth", ".ckpt", ".csv", ".dta", ".db", ".sqlite", ".sqlite3", ".pyc"}
PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "changeme",
    "example",
    "placeholder",
    "your_api_key",
    "your_api_key_here",
    "your_openai_api_key_here",
    "your_expected_parrot_api_key_here",
    "your_deepseek_api_key_here",
}
PLACEHOLDER_USERS = {"you", "username", "yourname"}
DOC_NETWORKS = (
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
)
GENERIC_ACCOUNTS = {"user", "admin", "ubuntu", "root", "test", "you", "username", "yourname"}
REMOVABLE = {TMP.name, BACKUP.name}

_USER = "|".join(re.escape(name) for name in sorted(PLACEHOLDER_USERS))
LOCAL_PATH = re.compile(rf"/Users/(?!({_USER})(?:/|$))[A-Za-z0-9._-]+(?:/[^\"'\n]*)?")
LINUX_HOME = re.compile(rf"/home/(?!({_USER})(?:/|$))[A-Za-z0-9._-]+(?:/[^\"'\n]*)?")
WIN_PROFILE = re.compile(
    rf"[A-Za-z]:\\+Users\\+(?!({_USER})(?:\\+|$))[^\\\"'\n]+(?:\\+[^\"'\n]*)?"
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
INTRANET_HOST = re.compile(
    r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+(?:local|internal|intra|corp|lan|home|localdomain)\b",
    re.IGNORECASE,
)
PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
CHINESE = re.compile(r"[\u4e00-\u9fff]")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?P<name>api[_-]?key|secret|access[_-]?token|password)\s*[:=]\s*['\"](?P<value>[^'\"]+)['\"]"
)
SECRET_TOKEN = re.compile(r"\b(?:sk-|ghp_|github_pat_|xox[baprs]-|AKIA)[A-Za-z0-9_\-]{16,}\b")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")
CREDENTIAL_URL = re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)
ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class PublishError(Exception):
    """A publish check failed before any snapshot was replaced."""


def git_output(args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            args,
            cwd=ROOT,
            input=stdin,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PublishError("git was not found") from exc
    except OSError as exc:
        raise PublishError(f"git failed: {exc}") from exc


def head_files() -> list[tuple[str, str]]:
    result = git_output(["git", "ls-tree", "-r", "-z", "HEAD"])
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PublishError(detail or "this directory is not a git repository")
    entries = []
    for item in result.stdout.split(b"\0"):
        if not item or b"\t" not in item:
            continue
        meta, path = item.split(b"\t", 1)
        mode = meta.split()[0].decode("ascii", "replace")
        relative = path.decode("utf-8", "surrogateescape")
        entries.append((mode, relative))
    return entries


def publish_tool_path(relative: str) -> bool:
    path = Path(relative)
    if path.name in SCRIPT_NAMES:
        return True
    if not path.parts:
        return False
    first = path.parts[0]
    return first in {DEST.name, TMP.name, BACKUP.name} or first.endswith("-pub")


def uncommitted_changes() -> list[str]:
    result = git_output(["git", "status", "--porcelain", "-z"])
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PublishError(detail or "git status failed")
    raw = result.stdout.decode("utf-8", "surrogateescape").split("\0")
    paths = []
    index = 0
    while index < len(raw):
        entry = raw[index]
        if not entry:
            index += 1
            continue
        status = entry[:2]
        path = entry[3:] if len(entry) > 3 else ""
        paths.append(path)
        index += 1
        if status[:1] in {"R", "C"} and index < len(raw):
            if raw[index]:
                paths.append(raw[index])
            index += 1
    return [path for path in paths if path and not publish_tool_path(path)]


def held_back(relative: str) -> bool:
    path = Path(relative)
    if path.name in SCRIPT_NAMES:
        return True
    candidate = ROOT / relative
    try:
        if candidate.exists() and candidate.resolve() == SCRIPT_PATH:
            return True
    except OSError:
        return True
    if path.parts and path.parts[0] in {DEST.name, TMP.name, BACKUP.name}:
        return True
    if ".git" in path.parts:
        return True
    return False


def local_account() -> str | None:
    name = getpass.getuser().strip()
    if len(name) < 4 or name.lower() in GENERIC_ACCOUNTS:
        return None
    return name


def credential_filename(relative: str) -> bool:
    name = Path(relative).name
    if name in ENV_EXAMPLE_NAMES:
        return False
    if name in CREDENTIAL_NAMES or (name.startswith(".env.") and name not in ENV_EXAMPLE_NAMES):
        return True
    return Path(name).suffix.lower() in CREDENTIAL_SUFFIXES


def data_artifact(relative: str) -> bool:
    path = Path(relative)
    if path.suffix.lower() in DATA_SUFFIXES:
        return True
    if not path.parts or path.parts[0] not in DATA_DIRECTORIES:
        return False
    return path.suffix.lower() not in {".py", ".pyi"}


def placeholder_secret(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_VALUES


def allowed_email(value: str) -> bool:
    return value.lower().rsplit("@", 1)[-1] in {"example.com", "example.org"}


def documentation_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(address in network for network in DOC_NETWORKS)


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def collect_matches(
    relative: str,
    label: str,
    pattern: re.Pattern[str],
    text: str,
    allow=None,
) -> list[str]:
    findings = []
    for match in pattern.finditer(text):
        if allow is not None and allow(match.group(0)):
            continue
        findings.append(f"{relative}:{line_number(text, match.start())}: {label}")
    return findings


def read_for_scan(path: Path) -> tuple[str | None, str | None]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PublishError(f"{path}: {exc}") from exc
    if b"\x00" in data[:8192]:
        return None, "binary file"
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "not utf-8"


def unquote_env(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def check_env_example(relative: str, text: str) -> list[str]:
    if Path(relative).name not in ENV_EXAMPLE_NAMES:
        return []
    findings = []
    for index, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_ASSIGNMENT.match(line)
        if match is None:
            continue
        name, value = match.group(1), unquote_env(match.group(2))
        if value and not placeholder_secret(value):
            findings.append(f"{relative}:{index}: env example value: {name}")
    return findings


def scan(relative: str, text: str) -> list[str]:
    findings: list[str] = []
    findings.extend(collect_matches(relative, "local path", LOCAL_PATH, text))
    findings.extend(collect_matches(relative, "home path", LINUX_HOME, text))
    findings.extend(collect_matches(relative, "Windows profile path", WIN_PROFILE, text))
    findings.extend(collect_matches(relative, "email address", EMAIL, text, allow=allowed_email))
    findings.extend(collect_matches(relative, "intranet host", INTRANET_HOST, text))
    findings.extend(collect_matches(relative, "private IP address", PRIVATE_IP, text, allow=documentation_ip))
    findings.extend(collect_matches(relative, "private key", PRIVATE_KEY, text))
    findings.extend(collect_matches(relative, "token", SECRET_TOKEN, text))
    if REJECT_CHINESE:
        for match in CHINESE.finditer(text):
            findings.append(f"{relative}:{line_number(text, match.start())}: Chinese text")
            break
    findings.extend(collect_matches(relative, "credential URL", CREDENTIAL_URL, text))
    findings.extend(notebook_findings(relative, text))
    for match in SECRET_ASSIGNMENT.finditer(text):
        if not placeholder_secret(match.group("value")):
            findings.append(
                f"{relative}:{line_number(text, match.start())}: hardcoded credential: {match.group('name')}"
            )
    findings.extend(account_context_findings(relative, text))
    findings.extend(check_env_example(relative, text))
    return findings


def account_context_findings(relative: str, text: str) -> list[str]:
    account = local_account()
    if not account:
        return []
    escaped = re.escape(account)
    patterns = (
        re.compile(rf"(?i)(?:/Users/|/home/){escaped}(?:/|$)"),
        re.compile(rf"(?i)(?<![A-Za-z0-9._%+\-]){escaped}@"),
        re.compile(rf"(?i)https?://[^/\s@]*{escaped}[^/\s@]*@"),
    )
    findings = []
    for pattern in patterns:
        findings.extend(collect_matches(relative, "account name", pattern, text))
    return findings


def notebook_findings(relative: str, text: str) -> list[str]:
    if not relative.endswith(".ipynb"):
        return []
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return [f"{relative}: notebook is not valid JSON"]
    findings = []
    for cell in document.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("outputs"):
            findings.append(f"{relative}: notebook outputs")
        if cell.get("execution_count") is not None:
            findings.append(f"{relative}: notebook execution count")
        if cell.get("attachments"):
            findings.append(f"{relative}: notebook attachments")
        if cell.get("metadata"):
            findings.append(f"{relative}: notebook cell metadata")
        if findings:
            break
    return findings


def repo_gitignore_pattern(meta: str) -> str | None:
    match = re.match(r"^(?P<source>.*?):(?P<line>\d+):(?P<pattern>.*)$", meta)
    if match is None:
        return None
    source = match.group("source")
    if source != ".gitignore" and not source.endswith("/.gitignore"):
        return None
    return match.group("pattern")


def pattern_ignores_env_file(pattern: str) -> bool:
    if pattern.endswith("/"):
        return False
    if pattern == ".env" or pattern.endswith("/.env"):
        return True
    # A dotfile glob covers .env in every directory it applies to.
    return pattern == ".*" or pattern.endswith("/.*")


def gitignore_ignores_env(tracked: list[str]) -> bool:
    """Both the repository root and a nested .env must be ignored by a project gitignore."""
    if ".gitignore" not in tracked:
        return False
    result = git_output(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--no-index",
            "-v",
            "--",
            *ENV_PATHS,
        ]
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PublishError(detail or "git check-ignore failed")
    ignored = set()
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if "\t" not in line:
            continue
        meta, pathname = line.split("\t", 1)
        pattern = repo_gitignore_pattern(meta)
        if pattern is not None and pattern_ignores_env_file(pattern):
            ignored.add(pathname)
    return set(ENV_PATHS) <= ignored


def ignored_paths(files: list[str]) -> list[str]:
    if not files:
        return []
    result = git_output(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--no-index",
            "-v",
            "-z",
            "--stdin",
        ],
        stdin="\0".join(files).encode("utf-8", "surrogateescape"),
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PublishError(detail or "git check-ignore failed")
    fields = result.stdout.decode("utf-8", "surrogateescape").split("\0")
    ignored = []
    for source, _line, pattern, pathname in zip(fields[0::4], fields[1::4], fields[2::4], fields[3::4]):
        if pattern.startswith("!"):
            continue
        if source == ".gitignore" or source.endswith("/.gitignore"):
            ignored.append(pathname)
    return ignored


def inside_root(path: Path) -> None:
    if path.is_symlink():
        raise PublishError(f"{path} is a symbolic link")
    try:
        resolved = path.resolve()
        root = ROOT.resolve()
    except OSError as exc:
        raise PublishError(f"{path}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PublishError(f"{path} resolves outside the repository") from exc


def validate_destination() -> None:
    if DEST.name != f"{PROJECT}-pub":
        raise PublishError(f"snapshot directory must be named {PROJECT}-pub")
    if DEST.is_symlink():
        raise PublishError(f"{DEST} is a symbolic link")
    if DEST.exists():
        inside_root(DEST)
    if TMP.is_symlink():
        raise PublishError(f"{TMP} is a symbolic link")
    if TMP.exists():
        inside_root(TMP)


def remove_tree(path: Path) -> None:
    if path.name not in REMOVABLE:
        raise PublishError(f"refusing to remove {path}")
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise PublishError(f"{path} is a symbolic link")
    inside_root(path)
    shutil.rmtree(path)


def export_head() -> None:
    validate_destination()
    if TMP.exists():
        remove_tree(TMP)
    TMP.mkdir()
    result = git_output(["git", "archive", "--format=tar", "HEAD"])
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PublishError(detail or "git archive failed")
    try:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            if hasattr(tarfile, "data_filter"):
                archive.extractall(TMP, filter="data")
            else:
                archive.extractall(TMP)
    except tarfile.TarError as exc:
        raise PublishError(f"git archive failed: {exc}") from exc
    extracted = list(TMP.rglob("*"))
    for path in extracted:
        if path.is_file() and not path.is_symlink() and path.name in SCRIPT_NAMES:
            path.unlink()
    for path in sorted(extracted, key=lambda item: len(item.parts), reverse=True):
        if not path.exists() and not path.is_symlink():
            continue
        relative = path.relative_to(TMP).as_posix()
        if not held_back(relative):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for path in TMP.rglob(".DS_Store"):
        if path.is_file() and not path.is_symlink():
            path.unlink()


def scan_tree(root: Path) -> list[str]:
    findings = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or ".git" in Path(relative).parts:
            findings.append(f"{relative}: symbolic link or git metadata")
            continue
        if path.name in SCRIPT_NAMES:
            findings.append(f"{relative}: excluded publisher file")
        if not path.is_file():
            continue
        if credential_filename(relative):
            findings.append(f"{relative}: credential filename")
        if data_artifact(relative):
            findings.append(f"{relative}: data artifact")
        text, problem = read_for_scan(path)
        if problem:
            findings.append(f"{relative}: {problem}")
            continue
        if text is not None:
            findings.extend(scan(relative, text))
    return findings


def promote_temp() -> None:
    validate_destination()
    if not TMP.is_dir() or TMP.is_symlink():
        raise PublishError(f"{TMP} is not a snapshot directory")
    if DEST.exists():
        if BACKUP.exists() or BACKUP.is_symlink():
            raise PublishError(f"{BACKUP} already exists")
        DEST.rename(BACKUP)
        try:
            TMP.rename(DEST)
        except OSError:
            if not DEST.exists():
                BACKUP.rename(DEST)
            raise
        remove_tree(BACKUP)
    else:
        TMP.rename(DEST)


def report(findings: list[str]) -> int:
    print("\nPre-publish check failed. Snapshot was not written.")
    for item in findings:
        print(f"  {item}")
    return 1


def main() -> int:
    try:
        dirty = uncommitted_changes()
    except PublishError as exc:
        print(exc)
        return 1
    if dirty:
        return report([f"uncommitted change: {path}" for path in dirty])

    try:
        entries = head_files()
    except PublishError as exc:
        print(exc)
        return 1

    tracked = [relative for _, relative in entries]
    selected = [relative for mode, relative in entries if mode != "120000" and not held_back(relative)]
    skipped = [relative for relative in tracked if held_back(relative)]
    print(f"Project: {PROJECT}")
    print(f"Snapshot: {DEST}")
    print(f"Files: {len(selected)}")
    if skipped:
        print("Held back:")
        for path in skipped:
            print(f"  {path}")

    findings: list[str] = []
    for name in REQUIRED_FILES:
        if name not in selected:
            findings.append(f"missing required file: {name}")
    for mode, relative in entries:
        if held_back(relative):
            continue
        if mode == "120000":
            findings.append(f"{relative}: symbolic link")
        if credential_filename(relative):
            findings.append(f"{relative}: credential filename")
        if data_artifact(relative):
            findings.append(f"{relative}: data artifact")
    try:
        if not gitignore_ignores_env(tracked):
            findings.append("missing gitignore rule: .env")
        for relative in ignored_paths(tracked):
            findings.append(f"{relative}: gitignored file")
    except PublishError as exc:
        print(exc)
        return 1

    if findings:
        return report(findings)

    try:
        export_head()
        missing = [
            relative
            for relative in selected
            if not (TMP / relative).is_file() or (TMP / relative).is_symlink()
        ]
        if missing:
            remove_tree(TMP)
            return report([f"{relative}: missing from snapshot" for relative in missing])
        written = scan_tree(TMP)
        if written:
            remove_tree(TMP)
            return report(written)
        promote_temp()
    except PublishError as exc:
        if TMP.exists() and not TMP.is_symlink():
            remove_tree(TMP)
        print(exc)
        return 1
    except OSError as exc:
        if TMP.exists() and not TMP.is_symlink():
            remove_tree(TMP)
        print(exc)
        return 1

    print(f"\nWrote {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
