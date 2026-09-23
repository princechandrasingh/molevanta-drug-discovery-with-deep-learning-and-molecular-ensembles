"""Collect license text from exact-version PyPI source archives, without executing them."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url):
    with urlopen(
        Request(url, headers={"User-Agent": "molstudy-license-audit/1"}), timeout=45
    ) as response:
        return response.read()


root = Path(__file__).resolve().parents[1] / "docs/license-audit"
packages = json.loads((root / "reference-inventory.json").read_text())["packages"]
records = []
for package in packages:
    if package["notices"]:
        continue
    name, version = package["name"], package["version"]
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    metadata = json.loads(fetch(metadata_url))
    source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    payload = fetch(source["url"])
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != source["digests"]["sha256"]:
        raise ValueError(f"Source checksum mismatch: {name}")
    notices = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or Path(member.name).name.lower() not in {
                "license",
                "license.rst",
                "license.txt",
                "licence",
                "licence.rst",
                "licence.txt",
                "copying",
                "notice",
            }:
                continue
            # Read only the notice bytes; never extract archive paths or execute source.
            text = archive.extractfile(member).read()
            relative = Path("source-notices") / f"{name}-{version}" / Path(member.name).name
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != text:
                raise ValueError(f"Conflicting notice: {relative}")
            target.write_bytes(text)
            notices.append(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(text).hexdigest(),
                    "archive_member": member.name,
                }
            )
    if not notices:
        raise ValueError(f"No source notice found for {name}")
    records.append(
        {
            "name": name,
            "version": version,
            "metadata_url": metadata_url,
            "source_url": source["url"],
            "source_sha256": checksum,
            "notices": notices,
        }
    )
    print(f"Collected {name} {version}: {len(notices)} source notice(s)", flush=True)
(root / "supplemental-notices.json").write_text(json.dumps(records, indent=2) + "\n")
