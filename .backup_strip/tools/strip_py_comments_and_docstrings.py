#!/usr/bin/env python3
"""
Strip Python `# ...` comments AND docstrings (triple-quoted blocks that are used
as module/class/function docstrings or standalone string statements).

This is intentionally destructive. It keeps:
  - shebang line (#!...) if present
  - encoding cookie (PEP 263) in first/second line if present
  - normal string literals used in assignments/expressions

It removes:
  - all COMMENT tokens
  - docstrings in modules, classes, and functions (AST docstrings)
  - standalone string statements (often used as multi-line comments)

Usage (from repo root):
  python tools/strip_py_comments_and_docstrings.py --root . --backup-dir .backup_strip
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path


ENCODING_RE = re.compile(r"^[ \t\f]*#.*coding[:=][ \t]*([-\w.]+)")


@dataclass(frozen=True)
class Span:
    start_line: int
    start_col: int
    end_line: int
    end_col: int


def _read_text(path: Path) -> str:
    # Preserve original newlines.
    return path.read_text(encoding="utf-8", errors="surrogateescape")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", errors="surrogateescape")


def _is_shebang(line: str) -> bool:
    return line.startswith("#!")


def _is_encoding_cookie(line: str) -> bool:
    return bool(ENCODING_RE.match(line))


def _docstring_spans(src: str) -> list[Span]:
    """
    Compute spans of docstrings and standalone string statements to delete.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    spans: list[Span] = []

    def add_docstring_span(node: ast.AST) -> None:
        # Only remove if the first statement is a constant string.
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            # Python 3.8+ has end_lineno / end_col_offset for constants in most cases.
            if getattr(first, "lineno", None) is None or getattr(first, "end_lineno", None) is None:
                return
            spans.append(
                Span(
                    start_line=int(first.lineno),
                    start_col=int(first.col_offset),
                    end_line=int(first.end_lineno),
                    end_col=int(first.end_col_offset),
                )
            )

    add_docstring_span(tree)  # module docstring
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_docstring_span(n)

    # Also remove any standalone string statements anywhere (often used as block comments).
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            if getattr(n, "lineno", None) is None or getattr(n, "end_lineno", None) is None:
                continue
            spans.append(
                Span(
                    start_line=int(n.lineno),
                    start_col=int(n.col_offset),
                    end_line=int(n.end_lineno),
                    end_col=int(n.end_col_offset),
                )
            )

    # Deduplicate (same span may be collected twice).
    uniq: dict[tuple[int, int, int, int], Span] = {}
    for s in spans:
        uniq[(s.start_line, s.start_col, s.end_line, s.end_col)] = s
    out = list(uniq.values())
    out.sort(key=lambda s: (s.start_line, s.start_col, s.end_line, s.end_col))
    return out


def _strip_comments_tokens(src: str) -> str:
    """
    Remove COMMENT tokens with tokenize, keeping spacing/newlines intact.
    """
    out_tokens: list[tokenize.TokenInfo] = []
    sio = StringIO(src)
    try:
        tokens = list(tokenize.generate_tokens(sio.readline))
    except tokenize.TokenError:
        # Fallback: crude line-based removal.
        return "\n".join([ln.split("#", 1)[0].rstrip() for ln in src.splitlines()]) + ("\n" if src.endswith("\n") else "")

    for t in tokens:
        if t.type == tokenize.COMMENT:
            continue
        out_tokens.append(t)
    return tokenize.untokenize(out_tokens)


def _apply_spans_delete(src: str, spans: list[Span]) -> str:
    """
    Delete given spans from source text.
    """
    if not spans:
        return src

    lines = src.splitlines(keepends=True)

    def idx(line_no: int, col: int) -> int:
        # line_no is 1-based.
        return sum(len(lines[i]) for i in range(0, line_no - 1)) + col

    # Work on absolute indices in the full string.
    deletes: list[tuple[int, int]] = []
    for s in spans:
        start = idx(s.start_line, s.start_col)
        end = idx(s.end_line, s.end_col)
        if end > start:
            deletes.append((start, end))

    # Merge overlapping deletes.
    deletes.sort()
    merged: list[tuple[int, int]] = []
    for a, b in deletes:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    out = []
    cur = 0
    for a, b in merged:
        out.append(src[cur:a])
        cur = b
    out.append(src[cur:])
    return "".join(out)


def strip_file(path: Path) -> bool:
    src = _read_text(path)
    original = src

    # Preserve shebang + encoding cookie if they exist; we’ll re-inject them after stripping.
    lines = src.splitlines(keepends=True)
    prefix: list[str] = []
    consumed = 0
    if lines:
        if _is_shebang(lines[0]):
            prefix.append(lines[0])
            consumed = 1
        if len(lines) > consumed and _is_encoding_cookie(lines[consumed]):
            prefix.append(lines[consumed])
            consumed += 1

    body = "".join(lines[consumed:])

    # Remove docstrings / standalone string statements first (AST spans are against body’s line numbers,
    # so parse the whole file and then delete on the full src).
    spans = _docstring_spans(src)
    src2 = _apply_spans_delete(src, spans)

    # Now remove `# ...` comments.
    src3 = _strip_comments_tokens(src2)

    # Re-ensure prefix (shebang/encoding) still present; if stripping removed them, re-add.
    # We only do this if the original had them.
    if prefix:
        new_lines = src3.splitlines(keepends=True)
        # Remove any existing shebang/encoding lines from the new content to avoid duplication.
        while new_lines and (_is_shebang(new_lines[0]) or _is_encoding_cookie(new_lines[0])):
            new_lines.pop(0)
        src3 = "".join(prefix) + "".join(new_lines)

    changed = src3 != original
    if changed:
        _write_text(path, src3)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root to process")
    ap.add_argument("--backup-dir", type=Path, default=Path(".backup_strip"), help="Backup dir (will be overwritten)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would change")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    backup = args.backup_dir.expanduser().resolve()

    py_files = [p for p in root.rglob("*.py") if ".venv" not in p.parts and "venv" not in p.parts]
    if not py_files:
        print("No .py files found.")
        return

    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True, exist_ok=True)

    changed_paths: list[Path] = []
    for p in py_files:
        rel = p.relative_to(root)
        b = backup / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, b)

        if args.dry_run:
            before = _read_text(p)
            # compute change without writing
            spans = _docstring_spans(before)
            after = _strip_comments_tokens(_apply_spans_delete(before, spans))
            if after != before:
                changed_paths.append(rel)
        else:
            if strip_file(p):
                changed_paths.append(rel)

    print(f"Backed up {len(py_files)} files to: {backup}")
    print(f"Changed {len(changed_paths)} files.")
    for rel in changed_paths[:50]:
        print(f"- {rel}")
    if len(changed_paths) > 50:
        print(f"... and {len(changed_paths) - 50} more")


if __name__ == "__main__":
    main()

