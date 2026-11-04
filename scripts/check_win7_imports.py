"""Checks that a Windows x64 program can start on Windows 7.

usage: python3 -I check_win7_imports.py <llvm-readobj> <program.exe> <win7-exports.txt>

Windows refuses to start a program whose import table names a DLL or a
function the system lacks ("The procedure entry point … could not be
located"). This lists the program's imports with llvm-readobj and looks each
one up in the export tables of Windows 7 (6.1.7600, x64) that YY-Thunks ships
as Config/x64/6.1.7600.txt (INI: [dll] sections of ordinal=name lines).
Delay-loaded imports are listed but allowed: they load only when called.
Exits with status 1 when an import is missing on Windows 7.
"""
import os
import re
import subprocess
import sys


def win7_exports(path):
    exports, dll = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            dll = exports.setdefault(line[1:-1].lower(), set())
        elif "=" in line and dll is not None:
            ordinal, name = line.split("=", 1)
            dll.add(name)
            dll.add(f"#{ordinal}")
    return exports


def main():
    readobj, exe, table = sys.argv[1:4]
    exports = win7_exports(table)
    text = subprocess.run([readobj, "--file-headers", "--coff-imports", exe], check=True, capture_output=True, text=True).stdout

    headers = dict(re.findall(r"^\s*(MajorSubsystemVersion|MinorSubsystemVersion): (\d+)", text, re.M))
    subsystem = (int(headers.get("MajorSubsystemVersion", 0)), int(headers.get("MinorSubsystemVersion", 0)))
    problems = []
    if subsystem > (6, 1):
        problems.append(f"PE subsystem version {subsystem[0]}.{subsystem[1]} (Windows 7 needs 6.1 or lower)")

    imports = []  # (dll, function or #ordinal, delay-loaded)
    dll, delayed = None, False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Import {") or line.startswith("DelayImport {"):
            dll, delayed = None, line.startswith("Delay")
        elif line.startswith("Name: ") and dll is None:
            dll = line[6:]
        elif line.startswith("Symbol: ") and dll:
            m = re.match(r"Symbol: (\S*) \((\d+)\)", line)
            name, ordinal = (m.group(1), m.group(2)) if m else (line[8:], "")
            imports.append((dll, name or f"#{ordinal}", delayed))

    for d, fn, delayed in imports:
        if delayed:
            continue
        have = exports.get(d.lower())
        if have is None:
            problems.append(f"{d}: this DLL does not exist on Windows 7")
        elif fn not in have:
            problems.append(f"{d}!{fn}: missing on Windows 7")

    dlls = sorted({d for d, _, _ in imports}, key=str.lower)
    lazy = sorted({d for d, _, delayed in imports if delayed}, key=str.lower)
    print(f"{os.path.basename(exe)}: {len(imports)} imports from {len(dlls)} DLLs, PE subsystem {subsystem[0]}.{subsystem[1]}")
    print("  DLLs: " + ", ".join(dlls))
    if lazy:
        print("  delay-loaded (allowed): " + ", ".join(lazy))
    if problems:
        print("NOT Windows 7 compatible:")
        for p in sorted(set(problems)):
            print("  " + p)
        sys.exit(1)
    print("Windows 7: every import exists")


main()
