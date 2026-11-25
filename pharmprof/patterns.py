"""Load, validate, and compile SMARTS pattern files."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import ChemicalFeatures

BUILTIN_NAMES = ("basefeatures", "pmapper", "nibr2019")

_TRUTHY = {"1", "true", "t", "yes", "y"}
_FALSY = {"0", "false", "f", "no", "n", ""}

_FAMILY_FROM_SHORT = {
    "a": "Aromatic",
    "d": "Donor",
    "hbd": "Donor",
    "a_acc": "Acceptor",
    "hba": "Acceptor",
    "p": "PosIonizable",
    "n": "NegIonizable",
    "h": "Hydrophobe",
    "x": "Halogen",
}


class PatternError(ValueError):
    """A pattern file could not be read or a SMARTS string would not compile."""


@dataclass
class Pattern:
    """One SMARTS pattern plus its metadata."""

    pattern_id: str
    name: str
    smarts: str
    family: str = ""
    role: str = "custom"
    source: str = ""
    license: str = ""
    citation: str = ""
    enabled: bool = True
    notes: str = ""
    query: Chem.Mol | None = field(default=None, repr=False, compare=False)

    def compiled(self) -> Chem.Mol:
        if self.query is None:
            raise PatternError(f"pattern {self.pattern_id!r} has no compiled query")
        return self.query


@dataclass
class PatternSet:
    """An ordered collection of patterns, typically one file or built-in set."""

    patterns: list[Pattern]
    source_path: str = ""

    def __iter__(self):
        return iter(self.patterns)

    def __len__(self) -> int:
        return len(self.patterns)

    def __getitem__(self, idx):
        return self.patterns[idx]

    def enabled(self) -> PatternSet:
        return PatternSet([p for p in self.patterns if p.enabled], self.source_path)

    def by_role(self, role: str) -> PatternSet:
        return PatternSet([p for p in self.patterns if p.role == role], self.source_path)

    def by_family(self, family: str) -> PatternSet:
        want = family.lower()
        return PatternSet(
            [p for p in self.patterns if p.family.lower() == want],
            self.source_path,
        )

    def ids(self) -> list[str]:
        return [p.pattern_id for p in self.patterns]

    def get(self, pattern_id: str) -> Pattern:
        for p in self.patterns:
            if p.pattern_id == pattern_id or p.name == pattern_id:
                return p
        raise KeyError(pattern_id)

    def extend(self, other: PatternSet | Sequence[Pattern]) -> PatternSet:
        extra = list(other.patterns) if isinstance(other, PatternSet) else list(other)
        seen = {p.pattern_id for p in self.patterns}
        merged = list(self.patterns)
        for p in extra:
            if p.pattern_id in seen:
                raise PatternError(f"duplicate pattern_id {p.pattern_id!r} when merging pattern sets")
            seen.add(p.pattern_id)
            merged.append(p)
        src = "+".join(s for s in (self.source_path, getattr(other, "source_path", "")) if s)
        return PatternSet(merged, src)


def compile_smarts(smarts: str) -> Chem.Mol:
    """Compile a SMARTS string, raising PatternError if RDKit cannot parse it."""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise PatternError(f"unparseable SMARTS: {smarts}")
    return mol


def _compile_or_raise(smarts: str, *, path: str, row: int, pattern_id: str, name: str) -> Chem.Mol:
    try:
        return compile_smarts(smarts)
    except PatternError as exc:
        where = f"{path} row {row}"
        ident = pattern_id or name or "(unnamed)"
        raise PatternError(
            f"unparseable SMARTS at {where} (pattern_id={ident}): {smarts}"
        ) from exc


def _parse_bool(value: str, *, path: str, row: int) -> bool:
    s = (value or "").strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    raise PatternError(f"invalid enabled flag {value!r} at {path} row {row}")


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return not s or s.startswith("#")


def _looks_like_header(fields: Sequence[str]) -> bool:
    lowered = [f.strip().lower() for f in fields]
    return "smarts" in lowered



def _unique_pattern_id(seen: set[str], pattern_id: str) -> str:
    if pattern_id not in seen:
        return pattern_id
    n = 2
    while f"{pattern_id}_{n}" in seen:
        n += 1
    return f"{pattern_id}_{n}"


def load_patterns(
    source: str | Path | Sequence[str | Path],
    *,
    include_disabled: bool = False,
) -> PatternSet:
    """Load one or more pattern sources.

    ``source`` may be a path to a TSV / two-column table / ``.fdef`` file, the
    name of a bundled set (``basefeatures``, ``pmapper``, ``nibr2019``), or a
    sequence of those. Disabled patterns are dropped unless
    ``include_disabled=True``.
    """
    if isinstance(source, (str, Path)):
        sources = [source]
    else:
        sources = list(source)
    combined = PatternSet([], "")
    for src in sources:
        combined = combined.extend(_load_one(src))
    return combined if include_disabled else combined.enabled()


def builtin_patterns(name: str = "basefeatures", *, include_disabled: bool = False) -> PatternSet:
    """Load a bundled pattern set by name."""
    return load_patterns(name, include_disabled=include_disabled)


def list_builtins() -> list[str]:
    return list(BUILTIN_NAMES)


def _load_one(source: str | Path) -> PatternSet:
    key = str(source).strip()
    if key in BUILTIN_NAMES:
        return _load_builtin(key)
    path = Path(key).expanduser()
    if not path.exists():
        raise PatternError(
            f"pattern source {key!r} is not a file and not a bundled set "
            f"({', '.join(BUILTIN_NAMES)})"
        )
    suffix = path.suffix.lower()
    if suffix == ".fdef":
        return _load_fdef(path)
    return _load_table(path)


def _bundled_file(name: str) -> Path:
    traversable = resources.files("pharmprof.pattern_data").joinpath(name)
    with resources.as_file(traversable) as path:
        return Path(path)


def _load_builtin(name: str) -> PatternSet:
    path = _bundled_file(f"{name}.tsv")
    return _load_table(path)


def builtin_fdef(name: str) -> Path:
    """Path to a bundled ``.fdef`` (``basefeatures`` or ``pmapper``)."""
    if name not in ("basefeatures", "pmapper"):
        raise PatternError(f"no bundled fdef named {name!r}")
    return _bundled_file(f"{name}.fdef")


def _load_table(path: Path) -> PatternSet:
    text = path.read_text(encoding="utf-8")
    content_lines: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        if _is_comment(line):
            continue
        content_lines.append((i, line))
    if not content_lines:
        raise PatternError(f"no patterns found in {path}")

    sample = next(csv.reader([content_lines[0][1]], delimiter="\t"))
    if _looks_like_header(sample):
        return _load_full_tsv(path, content_lines)
    return _load_two_column(path, content_lines)


def _load_full_tsv(path: Path, content_lines: list[tuple[int, str]]) -> PatternSet:
    header_row, header_line = content_lines[0]
    reader = csv.DictReader(
        [header_line] + [line for _, line in content_lines[1:]],
        delimiter="\t",
    )
    if not reader.fieldnames or "smarts" not in [f.lower() for f in reader.fieldnames]:
        raise PatternError(f"{path} header is missing a 'smarts' column")
    fieldmap = {f.lower(): f for f in reader.fieldnames}

    def col(row: dict, name: str, default: str = "") -> str:
        key = fieldmap.get(name)
        if key is None:
            return default
        return (row.get(key) or default).strip()

    patterns: list[Pattern] = []
    # DictReader iterates data rows; recover original line numbers from content_lines[1:]
    for (lineno, _), row in zip(content_lines[1:], reader, strict=False):
        smarts = col(row, "smarts")
        pattern_id = col(row, "pattern_id")
        name = col(row, "name") or pattern_id
        if not smarts:
            raise PatternError(f"empty SMARTS at {path} row {lineno}")
        if not pattern_id:
            pattern_id = name or f"{path.stem}.{len(patterns) + 1}"
        pattern_id = _unique_pattern_id({p.pattern_id for p in patterns}, pattern_id)
        enabled = _parse_bool(col(row, "enabled", "true"), path=str(path), row=lineno)
        query = None
        if enabled:
            query = _compile_or_raise(
                smarts, path=str(path), row=lineno, pattern_id=pattern_id, name=name
            )
        patterns.append(
            Pattern(
                pattern_id=pattern_id,
                name=name or pattern_id,
                smarts=smarts,
                family=col(row, "family"),
                role=col(row, "role", "custom") or "custom",
                source=col(row, "source"),
                license=col(row, "license"),
                citation=col(row, "citation"),
                enabled=enabled,
                notes=col(row, "notes"),
                query=query,
            )
        )
    if not patterns:
        raise PatternError(f"no patterns found in {path} (header at row {header_row})")
    return PatternSet(patterns, str(path))


def _which_is_smarts(a: str, b: str) -> tuple[str, str]:
    """Return (name, smarts) given two fields, using parse success as the cue."""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    a_ok = Chem.MolFromSmarts(a) is not None
    b_ok = Chem.MolFromSmarts(b) is not None
    if a_ok and not b_ok:
        return b, a
    if b_ok and not a_ok:
        return a, b
    if b_ok and a_ok:
        # Both parse (e.g. 'C' as a name). Prefer the conventional name, SMARTS order.
        return a, b
    # neither parses — caller will raise with context
    return a, b


def _load_two_column(path: Path, content_lines: list[tuple[int, str]]) -> PatternSet:
    patterns: list[Pattern] = []
    for lineno, line in content_lines:
        fields = next(csv.reader([line], delimiter="\t"))
        fields = [f.strip() for f in fields if f.strip() != ""]
        if len(fields) == 1:
            name, smarts = f"{path.stem}.{len(patterns) + 1}", fields[0]
        elif len(fields) == 2:
            name, smarts = _which_is_smarts(fields[0], fields[1])
        else:
            raise PatternError(
                f"{path} row {lineno} has {len(fields)} columns; "
                "expected a headered TSV or two columns (name, SMARTS)"
            )
        if not smarts:
            raise PatternError(f"empty SMARTS at {path} row {lineno}")
        pattern_id = _unique_pattern_id({p.pattern_id for p in patterns}, name or f"{path.stem}.{len(patterns) + 1}")
        query = _compile_or_raise(
            smarts, path=str(path), row=lineno, pattern_id=pattern_id, name=name
        )
        patterns.append(
            Pattern(
                pattern_id=pattern_id,
                name=name or pattern_id,
                smarts=smarts,
                role="custom",
                source=str(path),
                enabled=True,
                query=query,
            )
        )
    return PatternSet(patterns, str(path))


_FDEF_FEATURE = re.compile(r"DefineFeature\s+(\S+)\s+(.*)$", re.I)
_FDEF_FAMILY = re.compile(r"Family\s+(\S+)", re.I)


def _load_fdef(path: Path) -> PatternSet:
    """Load an RDKit ``.fdef`` via FeatureFactory, falling back to a text parse."""
    factory = ChemicalFeatures.BuildFeatureFactory(str(path))
    defs = factory.GetFeatureDefs()
    # GetFeatureDefs collapses duplicate names; parse the file so we keep them all.
    text_features = _parse_fdef_text(path)
    patterns: list[Pattern] = []
    used_ids: set[str] = set()
    if text_features:
        iterable: Iterable[tuple[str, str, str]] = text_features
    else:
        iterable = [(k.split(".", 1)[0], k.split(".", 1)[-1], sma) for k, sma in defs.items()]
    for family, name, smarts in iterable:
        base_id = f"{path.stem}.{name}"
        pattern_id = base_id
        n = 2
        while pattern_id in used_ids:
            pattern_id = f"{base_id}_{n}"
            n += 1
        used_ids.add(pattern_id)
        query = _compile_or_raise(
            smarts, path=str(path), row=0, pattern_id=pattern_id, name=name
        )
        patterns.append(
            Pattern(
                pattern_id=pattern_id,
                name=name if pattern_id.endswith(name) else pattern_id.split(".", 1)[-1],
                smarts=smarts,
                family=family,
                role="pharmacophore",
                source=str(path),
                enabled=True,
                query=query,
            )
        )
    if not patterns:
        raise PatternError(f"no DefineFeature blocks in {path}")
    return PatternSet(patterns, str(path))


def _parse_fdef_text(path: Path) -> list[tuple[str, str, str]]:
    features: list[tuple[str, str, str]] = []
    current: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        m = _FDEF_FEATURE.match(s)
        if m:
            current = {"name": m.group(1), "smarts": m.group(2).strip(), "family": ""}
            continue
        m = _FDEF_FAMILY.match(s)
        if m and current is not None:
            current["family"] = m.group(1)
            continue
        if re.match(r"EndFeature", s, re.I) and current is not None:
            smarts = current["smarts"]
            # Prefer expanded SMARTS from the factory when the raw string uses AtomTypes.
            features.append((current["family"], current["name"], smarts))
            current = None
    # Expand via factory when possible so AtomType macros compile.
    try:
        factory = ChemicalFeatures.BuildFeatureFactory(str(path))
        defs = factory.GetFeatureDefs()
    except Exception:
        defs = {}
    expanded: list[tuple[str, str, str]] = []
    seen_keys: dict[str, int] = {}
    for family, name, smarts in features:
        key = f"{family}.{name}" if family else name
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if seen_keys[key] == 1 and key in defs:
            smarts = defs[key]
        elif Chem.MolFromSmarts(smarts) is None and key in defs:
            smarts = defs[key]
        expanded.append((family, name, smarts))
    return expanded


def feature_factory(source: str | Path | None = None) -> ChemicalFeatures.MolChemicalFeatureFactory:
    """Build an RDKit feature factory from a bundled or user ``.fdef``."""
    if source is None or str(source) in {"basefeatures", ""}:
        path = builtin_fdef("basefeatures")
    elif str(source) == "pmapper":
        path = builtin_fdef("pmapper")
    else:
        path = Path(source)
    return ChemicalFeatures.BuildFeatureFactory(str(path))
