"""Ingestion: walk a directory of documents, chunk them, keep provenance.

Chunking is the most under-rated part of a RAG pipeline. STRATA chunks on
structural boundaries (markdown headings) first and only falls back to a sliding
window inside an over-long section, so a chunk almost always corresponds to one
coherent idea with its heading trail attached as context.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .text import tokenize

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

SKIP_DIRS = {
    ".git", "node_modules", ".next", "dist", "build", "__pycache__",
    ".venv", "venv", "Library", ".cache", "site-packages", ".claude",
}


@dataclass
class Chunk:
    id: int
    doc_id: str          # relative path of the source file
    title: str           # heading trail, e.g. "README > Architecture"
    text: str            # the chunk body (without the heading trail)
    start_line: int
    n_tokens: int
    project: str = ""    # top-level directory — useful as a facet

    def indexable(self) -> str:
        """What actually gets indexed: heading trail + body.

        The trail is repeated into the indexed text on purpose. A section titled
        "Hybrid scoring" whose body never repeats the phrase should still be
        retrievable by that phrase.
        """
        return f"{self.title}\n{self.text}"


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)
    root: str = ""

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> Chunk:
        return self.chunks[i]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                {"root": self.root, "chunks": [asdict(c) for c in self.chunks]},
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        with Path(path).open() as fh:
            blob = json.load(fh)
        return cls(
            root=blob["root"],
            chunks=[Chunk(**c) for c in blob["chunks"]],
        )


def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """Return (heading_trail, body, start_line) for each markdown section."""
    lines = text.splitlines()
    trail: list[str] = []
    sections: list[tuple[str, str, int]] = []
    buf: list[str] = []
    buf_start = 0
    buf_trail = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((buf_trail, body, buf_start + 1))

    in_fence = False
    for lineno, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            flush()
            depth = len(heading.group(1))
            title = heading.group(2).strip()
            trail = trail[: depth - 1]
            trail.append(title)
            buf, buf_start, buf_trail = [], lineno, " > ".join(trail)
        else:
            buf.append(line)
    flush()
    return sections


def _window(body: str, max_tokens: int, overlap: int) -> list[str]:
    """Sliding window over an over-long section, split on blank lines."""
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    out, current, current_len = [], [], 0
    for para in paragraphs:
        n = len(tokenize(para))
        if current and current_len + n > max_tokens:
            out.append("\n\n".join(current))
            # carry the tail of the previous window forward for context
            carry, carried = [], 0
            for prev in reversed(current):
                c = len(tokenize(prev))
                if carried + c > overlap:
                    break
                carry.insert(0, prev)
                carried += c
            current, current_len = carry, carried
        current.append(para)
        current_len += n
    if current:
        out.append("\n\n".join(current))
    return out or [body]


def build_corpus(
    roots: list[str],
    *,
    extensions: tuple[str, ...] = (".md", ".markdown", ".txt", ".mdx"),
    max_tokens: int = 320,
    overlap: int = 60,
    min_tokens: int = 20,
    max_files: int | None = None,
    exclude: tuple[str, ...] = (),
) -> Corpus:
    """Walk `roots`, chunk every matching file, return a Corpus.

    `exclude` holds case-insensitive substrings matched against each file's path
    — the practical way to keep vendored source dumps and generated output out
    of a documentation corpus.
    """
    exclude = tuple(e.lower() for e in exclude)
    corpus = Corpus(root=os.path.commonpath([os.path.abspath(r) for r in roots])
                    if len(roots) > 1 else os.path.abspath(roots[0]))
    next_id = 0
    files_seen = 0

    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if root_path.is_file():
            candidates = [root_path]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(root_path):
                dirnames[:] = [d for d in dirnames
                               if d not in SKIP_DIRS and not d.startswith(".")]
                for name in sorted(filenames):
                    if name.lower().endswith(extensions):
                        candidates.append(Path(dirpath) / name)

        for path in candidates:
            if max_files is not None and files_seen >= max_files:
                break
            if exclude and any(e in str(path).lower() for e in exclude):
                continue
            try:
                raw = path.read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            if not raw.strip():
                continue
            files_seen += 1

            try:
                rel = str(path.relative_to(root_path))
            except ValueError:
                rel = str(path)
            doc_id = f"{root_path.name}/{rel}" if root_path.is_dir() else path.name
            project = doc_id.split("/", 1)[0]
            doc_title = path.stem.replace("-", " ").replace("_", " ")

            for trail, body, line in _split_sections(raw):
                title = f"{doc_title} > {trail}" if trail else doc_title
                for piece in _window(body, max_tokens, overlap):
                    n = len(tokenize(piece))
                    if n < min_tokens:
                        continue
                    corpus.chunks.append(
                        Chunk(
                            id=next_id,
                            doc_id=doc_id,
                            title=title,
                            text=piece,
                            start_line=line,
                            n_tokens=n,
                            project=project,
                        )
                    )
                    next_id += 1
    return corpus
