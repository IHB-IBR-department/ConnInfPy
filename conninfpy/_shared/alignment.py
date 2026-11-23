"""Row alignment of paired conditions by opaque subject IDs.

Paired designs (two within-subject conditions, ``group1``/``group2`` or
``Y_A``/``Y_B``) only make sense when row ``s`` of condition B belongs to the
same subject as row ``s`` of condition A.  When the caller can name the
subjects, passing ``ids_group1`` / ``ids_group2`` lets the API verify that
bijection and reorder condition B instead of trusting the row order.

Contract
--------
* The two ID vectors must form a *strict full bijection*: the same unique,
  non-missing IDs on both sides, matched by Python equality of the opaque
  values.  Nothing is normalised, stringified or type-coerced, and IDs are
  never guessed from the data matrices.
* IDs are opaque hashable scalars (``str``, ``int``, ...).  ``None``, ``NaN``,
  ``NaT``, ``pd.NA`` and empty/whitespace-only strings are rejected as IDs.
* Condition A (``group1``/``Y_A``) is the reference: its row order *is* the
  result order.  One permutation moves every condition-B subject-axis input
  (condition matrices, condition-specific confounds, per-subject masks) into
  that order.
* Quantities that already live in A order (``interest``, subject-level
  ``confounds``, ``sites``, subject-level metadata) are validated against A
  and **never** permuted.  Edge-axis objects (``edge_mask``) are invariant
  under a subject permutation and are left untouched.
* Without IDs the previous positional behaviour is unchanged.

Index ownership, in one line per input group::

    condition A row order  ->  group1 / Y_A, interest, confounds, sites,
                               subject-level metadata   (reference, not moved)
    condition B row order  ->  group2 / Y_B, confounds_group2
                               (moved into A order)
    edge axis              ->  edge_mask, net_labels   (invariant)
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, MappingView, Set
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "IdAlignmentError",
    "validate_subject_ids",
    "paired_id_permutation",
    "reorder_subject_axis",
    "check_subject_axis",
    "alignment_record",
    "_MAX_LISTED_IDS",
]

#: How many offending IDs an error message spells out before truncating.
_MAX_LISTED_IDS = 5

#: Machine-readable reasons carried by :class:`IdAlignmentError`.
ID_ERROR_CODES = (
    "not_a_vector",  # the argument is not a sequence of IDs at all
    "invalid_id",    # missing / unhashable / non-scalar ID value
    "length",        # ID vector and subject axis disagree in length
    "duplicate",     # the same ID appears twice within one condition
    "one_sided",     # only one of the two conditions carries IDs
    "missing_pair",   # an ID has no partner in the other condition
    "not_paired",    # IDs supplied outside a paired design
)


class IdAlignmentError(ValueError):
    """A paired subject-ID vector cannot define a valid A↔B bijection.

    ``code`` names the failure (see :data:`ID_ERROR_CODES`) so that callers
    with their own error vocabulary (the YAML CLI raises ``ConfigError``) can
    translate it without parsing the message.  ``condition`` names the input
    that failed (``"group1"``/``"group2"``) when it is condition-specific and
    ``ids`` holds the offending ID values, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        condition: Optional[str] = None,
        ids: Sequence[Any] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.condition = condition
        self.ids = tuple(ids)


def _format_ids(ids: Sequence[Any]) -> str:
    """Comma-separated ``repr`` of at most :data:`_MAX_LISTED_IDS` IDs."""
    listed = ", ".join(repr(value) for value in list(ids)[:_MAX_LISTED_IDS])
    extra = len(ids) - _MAX_LISTED_IDS
    return listed + (f", … ({extra} more)" if extra > 0 else "")


def _is_missing_id(value: Any) -> bool:
    """True for ``None``/``NaN``/``NaT``/``pd.NA``/blank strings."""
    if value is None:
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return not value.strip()
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    return False


def validate_subject_ids(
    ids: Sequence[Any],
    name: str,
    *,
    n: Optional[int] = None,
    condition: Optional[str] = None,
) -> tuple:
    """Validate one condition's subject-ID vector and return it as a tuple.

    Parameters
    ----------
    ids : sequence of hashable scalars
        Subject IDs in the row order of that condition.
    name : str
        Argument name used in error messages (``'ids_group1'``,
        ``'ids_A'``, the CLI's subject column, ...).
    n : int, optional
        Length of the subject axis the IDs describe; checked when given.
    condition : str, optional
        Condition label (``'group1'``/``'group2'``) carried on the error.

    Returns
    -------
    tuple
        The IDs in input order, unchanged.

    Raises
    ------
    IdAlignmentError
        If the argument is not an *ordered* iterable of IDs (a string, an
        unordered set/view, a mapping, a scalar or any other non-iterable),
        contains a missing or unhashable ID, repeats an ID, or disagrees
        with ``n``.
    """
    if ids is None:
        raise IdAlignmentError(
            f"{name} is missing; pass IDs for both conditions or for neither "
            "(a one-sided ID vector cannot define the pairing).",
            code="one_sided",
            condition=condition,
        )
    if isinstance(ids, (str, bytes, bytearray)):
        raise IdAlignmentError(
            f"{name} must be a sequence of subject IDs, not a single string "
            f"({ids!r}).",
            code="not_a_vector",
            condition=condition,
        )
    # Row order is the whole point of an ID vector: an unordered collection
    # would pair rows by an arbitrary iteration order, and a mapping has no
    # row order at all.  Both are rejected instead of silently guessed.
    if isinstance(ids, Mapping):
        raise IdAlignmentError(
            f"{name} must be a sequence of subject IDs in row order, not a "
            f"mapping ({type(ids).__name__}); a mapping carries no row order "
            "to pair on — pass the IDs as a list, array or pandas Series.",
            code="not_a_vector",
            condition=condition,
        )
    if isinstance(ids, Set):
        raise IdAlignmentError(
            f"{name} must be a sequence of subject IDs in row order, not an "
            f"unordered set ({type(ids).__name__}); a set has no row order, so "
            "it cannot say which row belongs to which subject.",
            code="not_a_vector",
            condition=condition,
        )
    if isinstance(ids, MappingView):
        raise IdAlignmentError(
            f"{name} must be a sequence of subject IDs in row order, not an "
            f"unordered view ({type(ids).__name__}); pass the IDs as a list, "
            "array or pandas Series.",
            code="not_a_vector",
            condition=condition,
        )
    try:
        iterator = iter(ids)
    except TypeError:
        raise IdAlignmentError(
            f"{name} must be a sequence of subject IDs, not a single "
            f"non-iterable value ({ids!r}); pass one ID per subject row.",
            code="not_a_vector",
            condition=condition,
        ) from None
    values = list(iterator)
    for position, value in enumerate(values):
        if _is_missing_id(value):
            raise IdAlignmentError(
                f"{name}[{position}] is a missing subject ID "
                f"({value!r}); paired IDs must be non-empty and non-missing, "
                "and no subject may be silently dropped or filled in.",
                code="invalid_id",
                condition=condition,
            )
        if not isinstance(value, Hashable):
            raise IdAlignmentError(
                f"{name}[{position}] is not a hashable scalar "
                f"({value!r}); subject IDs must be opaque hashable values "
                "such as str or int.",
                code="invalid_id",
                condition=condition,
            )
    seen: dict = {}
    for position, value in enumerate(values):
        if value in seen:
            raise IdAlignmentError(
                f"{name} repeats subject ID {value!r} at positions "
                f"{seen[value]} and {position}; each subject needs exactly one "
                "row per condition.",
                code="duplicate",
                condition=condition,
                ids=(value,),
            )
        seen[value] = position
    if n is not None and len(values) != n:
        raise IdAlignmentError(
            f"{name} has {len(values)} subject IDs but the matching array has "
            f"{n} rows on the subject axis.",
            code="length",
            condition=condition,
        )
    return tuple(values)


def paired_id_permutation(
    ids_group1: Sequence[Any],
    ids_group2: Sequence[Any],
    *,
    n_group1: Optional[int] = None,
    n_group2: Optional[int] = None,
    name_group1: str = "ids_group1",
    name_group2: str = "ids_group2",
) -> npt.NDArray[np.intp]:
    """Row permutation that puts condition B into condition A's order.

    Returns
    -------
    ndarray of intp, shape ``(n_subjects,)``
        ``group2[perm]`` (and every other condition-B subject-axis input)
        lines up with ``group1``.  Condition A is the reference and is never
        permuted.

    Raises
    ------
    IdAlignmentError
        If either vector is invalid (see :func:`validate_subject_ids`), the
        two vectors are not a full bijection (an ID is missing from one
        condition), or a vector disagrees with its array's subject axis.
    """
    a = validate_subject_ids(
        ids_group1, name_group1, n=n_group1, condition="group1"
    )
    b = validate_subject_ids(
        ids_group2, name_group2, n=n_group2, condition="group2"
    )
    position_in_b = {value: position for position, value in enumerate(b)}
    in_a = set(a)
    missing_in_b = [value for value in a if value not in position_in_b]
    missing_in_a = [value for value in b if value not in in_a]
    if missing_in_b or missing_in_a:
        details = []
        if missing_in_b:
            details.append(
                f"not in {name_group2}: {_format_ids(missing_in_b)}"
            )
        if missing_in_a:
            details.append(
                f"not in {name_group1}: {_format_ids(missing_in_a)}"
            )
        raise IdAlignmentError(
            "paired subject IDs must form a strict one-to-one match between "
            "the two conditions; ID(s) present in only one condition — "
            + "; ".join(details)
            + ". Every subject needs both conditions; no observation is "
            "dropped or duplicated silently.",
            code="missing_pair",
            ids=tuple(missing_in_b) + tuple(missing_in_a),
        )
    return np.array([position_in_b[value] for value in a], dtype=np.intp)


def check_subject_axis(
    values: Optional[Any],
    name: str,
    n: int,
    *,
    condition: Optional[str] = None,
) -> None:
    """Require ``values`` to carry exactly ``n`` rows (A-order validation).

    Used for inputs that already live in condition A's row order
    (``interest``, subject-level ``confounds``, ``sites``): they are checked
    against A and never permuted.
    """
    if values is None:
        return
    arr = np.asarray(values)
    if arr.ndim == 0 or arr.shape[0] != n:
        rows = "a scalar" if arr.ndim == 0 else f"{arr.shape[0]} rows"
        raise IdAlignmentError(
            f"{name} has {rows} on the subject axis but the reference "
            f"condition has {n} subjects; A-order inputs must be given in "
            "condition A's row order and are never permuted.",
            code="length",
            condition=condition,
        )


def reorder_subject_axis(
    values: Optional[Any],
    perm: npt.NDArray[np.intp],
    name: str,
    *,
    condition: Optional[str] = "group2",
) -> Optional[Any]:
    """Move a condition-B subject-axis input into condition A's order."""
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 0 or arr.shape[0] != perm.shape[0]:
        rows = "a scalar" if arr.ndim == 0 else f"{arr.shape[0]} rows"
        raise IdAlignmentError(
            f"{name} has {rows} on the subject axis but the condition-B ID "
            f"vector describes {perm.shape[0]} subjects.",
            code="length",
            condition=condition,
        )
    return arr[perm]


def alignment_record(
    perm: npt.NDArray[np.intp],
    *,
    n_subjects: int,
    reference: str = "group1",
    reordered_inputs: Sequence[str] = (),
    a_order_inputs: Sequence[str] = (),
) -> dict:
    """Provenance record of one applied ID alignment (JSON-friendly).

    ``permutation_b_to_a[k]`` is the condition-B row placed at reference row
    ``k``; ``identity_permutation`` says whether the caller's B rows were
    already in A order.  Stored under ``result.design['paired_alignment']``
    so it survives :meth:`InferenceResult.to_bundle`.
    """
    perm = np.asarray(perm, dtype=np.intp)
    identity = bool(np.array_equal(perm, np.arange(perm.shape[0], dtype=np.intp)))
    return {
        "applied": True,
        "method": "subject_id_bijection",
        "reference": reference,
        "n_subjects": int(n_subjects),
        "identity_permutation": identity,
        "permutation_b_to_a": [int(index) for index in perm],
        "reordered_inputs": list(reordered_inputs),
        "a_order_inputs_validated_only": list(a_order_inputs),
        "index_ownership": (
            "condition A row order is the result order; condition-B subject-axis "
            "inputs were permuted into it; interest, subject-level confounds, "
            "sites and subject-level metadata stay in A order; edge-axis masks "
            "are invariant."
        ),
    }
