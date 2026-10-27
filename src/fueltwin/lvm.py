"""Minimal reader for LabVIEW Measurement (.lvm) text files.

Both data sources in this project are .lvm files, but with different layouts:

* 2014 fuel-rig files: 13 channels at 1 kHz with real time stamps.
* 2021/2023 pump-test files: 29 PLC channels at 200 ms, time column all zeros.

The number of header lines is not fixed, so the reader looks for the
``X_Value`` row that holds the channel names rather than skipping a fixed count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class LvmFile:
    path: Path
    header: dict[str, str] = field(default_factory=dict)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


def _find_column_row(path: Path) -> tuple[int, list[str], dict[str, str]]:
    header: dict[str, str] = {}
    with open(path, encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] == "X_Value":
                names = [p for p in parts if p != ""]
                return i, names, header
            # File-level header keys keep their first occurrence. The per-channel
            # block repeats Date/Time with one value per channel; its first value
            # is the actual start of this recording, kept as channel_<key>.
            if len(parts) >= 2 and parts[0]:
                key = parts[0]
                if len([p for p in parts[1:] if p]) > 1 and f"channel_{key}" not in header:
                    header[f"channel_{key}"] = parts[1]
                elif key not in header:
                    header[key] = parts[1]
    raise ValueError(f"No 'X_Value' column row found in {path}")


def read_lvm(path: str | Path) -> LvmFile:
    """Read an .lvm file into a DataFrame with the file's own channel names.

    A trailing ``Comment`` column (always empty here) is dropped.
    """
    path = Path(path)
    row, names, header = _find_column_row(path)
    if names and names[-1] == "Comment":
        names = names[:-1]
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=row + 1,
        header=None,
        usecols=range(len(names)),
        names=names,
        engine="c",
        dtype="float64",
    )
    return LvmFile(path=path, header=header, data=df)
