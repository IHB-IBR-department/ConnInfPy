"""Three-state preprocessing facts carried with a dataset and its results.

Every fact is either **unknown** (absent / ``None``) or known with an
explicit *source* that separates what ConnInfPy itself did from what an
input manifest or file name merely declares:

* ``"computed_by_conninfpy"`` — the connectivity backend
  (:func:`conninfpy.compute_connectivity`) estimated the matrices.
* ``"loader_denoising_pipeline"`` — the loader's own denoising (e.g.
  fMRIPrep confound regression incl. its global-signal option).
* ``"input_file_name_token"`` — a declaration encoded in the input file
  naming convention (e.g. ``..._GSR.npy``); not performed by this library.
* ``"input_manifest_declared"`` — declared in the dataset manifest's
  ``preprocessing.applied`` block about upstream processing.
* ``"app_rank_correlation_pipeline"`` — the app ranked the series and the
  backend correlated the ranks (Spearman).

A missing key or a ``None`` fact means *unknown*, never "not applied":
an explicit ``False`` (e.g. global-signal regression known to be off) is
kept as ``{"value": False, "source": ...}`` and must not collapse to
missing. Known facts are never inferred from the connectivity values
themselves (GSR is not guessed from negative edges).

Per-observation diagnostics live in the phenotype table, not here:
``n_valid_tr`` counts retained frames and ``delta_s`` is a per-subject
shrinkage intensity in [0, 1] — neither is a repetition time.
``tr_seconds`` is only ever recorded from an explicit source; absent
means unknown, not 1 or any other default.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Optional

PREPROCESSING_SCHEMA_VERSION = 1

SOURCE_COMPUTED = "computed_by_conninfpy"
SOURCE_LOADER_DENOISING = "loader_denoising_pipeline"
SOURCE_FILE_TOKEN = "input_file_name_token"
SOURCE_MANIFEST = "input_manifest_declared"
SOURCE_APP_RANKS = "app_rank_correlation_pipeline"

#: Connectivity estimators that can be stated as known facts.
KNOWN_CONNECTIVITY_METHODS = ("pearson", "partial", "spearman")

_FACT_KEYS = ("gsr", "connectivity_method", "standardize", "shrinkage", "tr_seconds")
#: Supplementary per-constituent detail (e.g. per-site records combined by a
#: multi-site loader). It never participates in the global facts: a fact the
#: constituents disagree on stays globally unknown while ``per_site`` keeps
#: each constituent's known value and source.
_PER_SITE_KEY = "per_site"
_SHRINKAGE_NAMES = ("ledoit_wolf", "oas")


def fact(value: Any, source: str) -> dict:
    """Build one known fact: ``{"value": ..., "source": ...}``."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("a preprocessing fact needs a nonempty source string")
    return {"value": deepcopy(value), "source": source}


def connectivity_method_from_kind(kind: Any) -> str:
    """Map backend ``kind`` names onto fact values (``spearman`` passes through)."""
    kind = str(kind)
    if kind in ("correlation", "pearson"):
        return "pearson"
    if kind == "partial_correlation":
        return "partial"
    if kind == "spearman":
        return "spearman"
    raise ValueError(f"unknown connectivity kind {kind!r}")


def gsr_from_file_token(token: Any) -> Optional[bool]:
    """Global-signal status declared by a file-name token, or ``None``.

    ``"GSR"`` (case-insensitive, punctuation ignored) declares that
    global-signal regression was applied upstream; ``"noGSR"`` / ``"no_GSR"``
    / ``"no-GSR"`` declare that it was not. Any other token declares
    nothing (``None`` = unknown).
    """
    if not isinstance(token, str):
        return None
    normalized = "".join(ch for ch in token.upper() if ch.isalnum())
    if normalized == "GSR":
        return True
    if normalized in ("NOGSR", "WITHOUTGSR", "NOGLOBALSIGNALREGRESSION"):
        return False
    return None


def gsr_from_file_name(path: Any) -> Optional[bool]:
    """Global-signal status declared by a file *name* (not its contents).

    Scans the base name for the ``GSR`` / negated-``GSR`` tokens only; a
    name without a supported token declares nothing. Negation survives
    underscores, hyphens, dots and case (``no_GSR``, ``no-GSR``, ``noGSR``,
    ``no.gsr``), and a negated token never reads as a plain ``GSR`` token;
    a ``GSR`` embedded in a larger word (e.g. ``xgsr``) declares nothing.
    """
    import re

    if not isinstance(path, (str, Path)):
        return None
    tokens = re.findall(r"[A-Za-z0-9]+", Path(path).name.upper())
    # Negation first: a standalone negated token, or the token pair NO + GSR
    # (covers no_GSR / no-GSR / no.GSR across separators).
    if any(gsr_from_file_token(token) is False for token in tokens):
        return False
    if any(tok == "NO" and nxt == "GSR" for tok, nxt in zip(tokens, tokens[1:])):
        return False
    if "GSR" in tokens:
        return True
    return None


def empty_record() -> dict:
    """A record with every fact unknown."""
    return {"schema_version": PREPROCESSING_SCHEMA_VERSION}


def _validate_fact(key: str, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {"value", "source"}:
        raise ValueError(f"preprocessing fact {key!r} must be {'value', 'source'}")
    source = value["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"preprocessing fact {key!r} needs a nonempty source string")
    v = value["value"]
    if key == "gsr" and not isinstance(v, bool):
        raise ValueError("gsr value must be a boolean")
    if key == "connectivity_method" and v not in KNOWN_CONNECTIVITY_METHODS:
        raise ValueError(f"connectivity_method value must be one of {KNOWN_CONNECTIVITY_METHODS}")
    if key == "standardize" and not isinstance(v, bool):
        raise ValueError("standardize value must be a boolean")
    if key == "shrinkage":
        if v is not None and not (v in _SHRINKAGE_NAMES or
                                  (isinstance(v, Real) and not isinstance(v, bool)
                                   and 0.0 <= float(v) <= 1.0)):
            raise ValueError("shrinkage value must be None, 'ledoit_wolf', 'oas', or a float in [0, 1]")
    if key == "tr_seconds":
        if not (isinstance(v, Real) and not isinstance(v, bool) and 0 < float(v) < float("inf")):
            raise ValueError("tr_seconds value must be a positive finite number")


def _normalize_record(record: Any, *, allow_per_site: bool) -> dict:
    if record is None:
        return empty_record()
    if not isinstance(record, Mapping):
        raise ValueError("the preprocessing record must be a mapping or None")
    version = record.get("schema_version", PREPROCESSING_SCHEMA_VERSION)
    if version != PREPROCESSING_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported preprocessing record schema_version {version!r}; "
            f"supported: {PREPROCESSING_SCHEMA_VERSION}"
        )
    allowed = set(_FACT_KEYS) | {_PER_SITE_KEY} if allow_per_site else set(_FACT_KEYS)
    unknown = set(record) - allowed - {"schema_version"}
    if unknown:
        raise ValueError(f"unknown preprocessing fact(s): {sorted(unknown)}")
    out: dict = {"schema_version": PREPROCESSING_SCHEMA_VERSION}
    for key in _FACT_KEYS:
        value = record.get(key)
        if value is None:
            continue
        _validate_fact(key, value)
        out[key] = {"value": deepcopy(value["value"]), "source": str(value["source"])}
    per_site = record.get(_PER_SITE_KEY)
    if per_site is not None:
        if not allow_per_site:
            raise ValueError("per_site records cannot be nested")
        if not isinstance(per_site, Mapping):
            raise ValueError("per_site must be a mapping of site name to record")
        sites: dict = {}
        for site, site_record in per_site.items():
            if not isinstance(site, str) or not site.strip():
                raise ValueError("per_site keys must be nonempty site names")
            sites[site] = _normalize_record(site_record, allow_per_site=False)
        out[_PER_SITE_KEY] = sites
    return out


def normalize_preprocessing_record(record: Any) -> dict:
    """Validate (or accept ``None`` for) a preprocessing record; return a copy.

    Unknown keys are rejected so typos cannot silently hide facts; every
    known fact must carry a value and a source. An explicit ``schema_version``
    other than the supported one is rejected rather than silently rewritten
    (a future schema may mean something different); an absent version reads
    as the current one, as loaders building records programmatically do.
    The optional ``per_site`` mapping carries per-constituent records (e.g.
    per-site facts of a combined multi-site dataset) alongside the global
    facts; site records may not nest ``per_site`` again.
    """
    return _normalize_record(record, allow_per_site=True)


def merge_preprocessing_records(*records: Any) -> dict:
    """Merge records; earlier arguments win (library-computed beats declared).

    Only facts missing from the higher-precedence record are filled in, so
    an explicit ``False`` from a performed pipeline is never overwritten by
    a manifest declaration.
    """
    merged = empty_record()
    for record in records:
        if record is None:
            continue
        normalized = normalize_preprocessing_record(record)
        for key, value in normalized.items():
            if key == "schema_version":
                continue
            merged.setdefault(key, value)
    return merged


def aggregate_preprocessing_records(records: Any, *, names: Any = None) -> dict:
    """Aggregate per-constituent records (e.g. per-site) into one record.

    A fact is asserted globally only when **every** constituent states it
    with the same value: differing values, or a mix of known and unknown,
    leave the global fact unknown — never an invented common True/False.
    The aggregated source is the constituents' shared source, or a
    ``mixed_sources(...)`` description when the agreeing facts arrived via
    different sources. Per-constituent records stay available to the caller
    (e.g. as ``provenance['preprocessing_per_site']``) so heterogeneity is
    inspectable; ``names`` only labels the constituents.
    """
    if isinstance(records, Mapping):
        pairs = list(records.items())
    else:
        records_list = list(records)
        labels = list(names) if names is not None else [str(i) for i in range(len(records_list))]
        pairs = list(zip(labels, records_list))
    if not pairs:
        return empty_record()
    normalized = [(str(name), normalize_preprocessing_record(record)) for name, record in pairs]
    aggregated = empty_record()
    for key in _FACT_KEYS:
        knowing = [(name, record[key]) for name, record in normalized if key in record]
        if not knowing or len(knowing) != len(normalized):
            continue  # unknown somewhere (or everywhere) → globally unknown
        if len({repr(value["value"]) for _, value in knowing}) != 1:
            continue  # conflicting statements → globally unknown
        sources = {value["source"] for _, value in knowing}
        source = next(iter(sources)) if len(sources) == 1 else \
            "mixed_sources(" + "+".join(sorted(sources)) + ")"
        aggregated[key] = {"value": deepcopy(knowing[0][1]["value"]), "source": source}
    return aggregated


def attach_preprocessing(dataset: Any, **facts: Any) -> dict:
    """Attach/extend ``dataset.provenance['preprocessing']`` in place.

    Each keyword is one fact value paired with its source either directly
    as a ``('value', 'source')`` tuple or via ``fact(value, source)``;
    ``None`` facts are ignored. Facts already recorded on the dataset win
    (they were set by the component that actually did the work).
    """
    record = merge_preprocessing_records(
        dataset.provenance.get("preprocessing"),
        {key: (fact(*value) if isinstance(value, tuple) else value) for key, value in facts.items()},
    )
    dataset.provenance["preprocessing"] = record
    return record


def set_preprocessing_facts(dataset: Any, **facts: Any) -> dict:
    """Overwrite facts recorded on a dataset (e.g. Spearman rank correlation).

    Same calling convention as :func:`attach_preprocessing`, but the given
    facts replace whatever was recorded before, because the caller states
    what the data actually are after its own processing step.
    """
    existing = dataset.provenance.get("preprocessing") or empty_record()
    updates = {key: (fact(*value) if isinstance(value, tuple) else value) for key, value in facts.items()}
    record = normalize_preprocessing_record({**existing, **updates})
    dataset.provenance["preprocessing"] = record
    return record


def record_from_manifest_applied(applied: Any) -> dict:
    """Facts declared by a manifest ``preprocessing.applied`` block.

    Recognised keys (everything else in the block is left alone, it stays
    available in the manifest itself): ``use_GSR``/``gsr`` (bool),
    ``correlation``/``connectivity_method`` (pearson/partial/spearman),
    ``standardize`` (bool), ``shrinkage`` (null/'ledoit_wolf'/'oas'/float),
    ``tr_seconds`` (positive number). Only declarations the block actually
    makes become facts; nothing is defaulted.
    """
    if not isinstance(applied, Mapping):
        return empty_record()
    record = empty_record()
    gsr = applied.get("use_GSR", applied.get("gsr"))
    if isinstance(gsr, bool):
        record["gsr"] = fact(gsr, SOURCE_MANIFEST)
    method = applied.get("correlation", applied.get("connectivity_method"))
    if isinstance(method, str) and method in KNOWN_CONNECTIVITY_METHODS:
        record["connectivity_method"] = fact(method, SOURCE_MANIFEST)
    standardize = applied.get("standardize")
    if isinstance(standardize, bool):
        record["standardize"] = fact(standardize, SOURCE_MANIFEST)
    shrinkage = applied.get("shrinkage", _OMITTED)
    if shrinkage is not _OMITTED:
        record["shrinkage"] = fact(_coerce_shrinkage(shrinkage), SOURCE_MANIFEST)
    tr = applied.get("tr_seconds")
    if isinstance(tr, Real) and not isinstance(tr, bool) and 0 < float(tr) < float("inf"):
        record["tr_seconds"] = fact(float(tr), SOURCE_MANIFEST)
    return record


_OMITTED = object()


def _coerce_shrinkage(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value in _SHRINKAGE_NAMES:
        return value
    if isinstance(value, Real) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0:
        return float(value)
    raise ValueError(f"invalid declared shrinkage {value!r}")


__all__ = [
    "PREPROCESSING_SCHEMA_VERSION",
    "SOURCE_COMPUTED",
    "SOURCE_LOADER_DENOISING",
    "SOURCE_FILE_TOKEN",
    "SOURCE_MANIFEST",
    "SOURCE_APP_RANKS",
    "KNOWN_CONNECTIVITY_METHODS",
    "fact",
    "connectivity_method_from_kind",
    "gsr_from_file_token",
    "gsr_from_file_name",
    "empty_record",
    "normalize_preprocessing_record",
    "merge_preprocessing_records",
    "aggregate_preprocessing_records",
    "attach_preprocessing",
    "set_preprocessing_facts",
    "record_from_manifest_applied",
]
