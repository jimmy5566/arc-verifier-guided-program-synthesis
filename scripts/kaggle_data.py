"""Safely check/download data without exposing credentials."""
from __future__ import annotations
import subprocess, zipfile, sys
from pathlib import Path
COMPETITION="arc-prize-2026-arc-agi-2"
raw=Path("data/raw"); raw.mkdir(parents=True,exist_ok=True)
kaggle_command=[sys.executable,"-m","kaggle"]
check=subprocess.run(kaggle_command+["competitions","files","-c",COMPETITION],capture_output=True,text=True)
if check.returncode:
    text=(check.stderr+check.stdout).lower()
    if "rule" in text or "join" in text or "accept" in text: raise SystemExit("请在 Kaggle ARC-AGI-2 页面点击 Join Competition 并接受 Rules。")
    raise SystemExit("Kaggle competition access failed without printing credentials. Check `kaggle.json` setup and competition access.")
print(check.stdout)
download=subprocess.run(kaggle_command+["competitions","download","-c",COMPETITION,"-p",str(raw)],text=True)
if download.returncode: raise SystemExit(download.returncode)
archives=sorted(raw.glob("*.zip"))
if not archives: raise SystemExit("Kaggle download completed but no archive was found.")
with zipfile.ZipFile(archives[-1]) as archive:
    for member in archive.infolist():
        target=(raw/member.filename).resolve()
        if not target.is_relative_to(raw.resolve()): raise SystemExit("Refusing unsafe archive member path")
    archive.extractall(raw)
print("Downloaded and extracted:")
for path in sorted(raw.glob("*.json")): print(f"{path.name}\t{path.stat().st_size} bytes")
