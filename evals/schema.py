"""Frozen dataset schemas for the Phase 0 single-turn experiment.

Every dataset the sweep reads is a JSONL file under ``evals/data/``, generated
once by ``evals.build_datasets`` and then treated as immutable raw material.
This module is the only place that knows what a row means.

Two properties are enforced here rather than by discipline:

*A contrast set that violates a matching invariant cannot be used.* Difference
in means will happily fit topic, length, or punctuation if a pair differs in
those. ``validate_contrast_set`` refuses a set whose pairs are not matched on
**rendered** length (PREFLIGHT.md §5), whose scenario families straddle the
train/held-out split, or whose ids are not unique. A validation failure stops
the pipeline; it is never downgraded to a warning.

*A dataset is identified by the hash of its content.* Every run manifest
records the hash of every dataset it read, so a result can always be traced to
the exact rows that produced it. Hashing reuses ``directions.hash_contrasts``
so that a contrast set hashes identically here and in a direction bundle.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from directions.bundle import canonical_contrast_line, hash_contrasts

__all__ = [
    "DATA_ROOT",
    "SPLITS",
    "DatasetError",
    "ContrastPair",
    "InjectionItem",
    "RetainItem",
    "StructuredItem",
    "CapabilityItem",
    "SafetyItem",
    "read_jsonl",
    "write_jsonl",
    "hash_rows",
    "hash_file",
    "load_contrast_set",
    "validate_contrast_set",
    "load_injection_set",
    "load_retain_set",
    "load_structured_set",
    "load_capability_set",
    "load_safety_set",
    "rendered_tool_content",
]

DATA_ROOT = Path(__file__).resolve().parent / "data"

SPLITS = ("train", "heldout")

# Rendered tool-content length tolerance within a pair, as a fraction of the
# longer member. PREFLIGHT.md §5: pairs must be matched on rendered length
# after JSON escaping, not raw content length, because the template escapes
# tool content and an unmatched pair lets difference-in-means fit length.
LENGTH_TOLERANCE = 0.12

# Absolute floor, so that very short pairs are not rejected for a difference of
# a few characters that the fractional tolerance would forbid.
LENGTH_TOLERANCE_FLOOR = 12


class DatasetError(ValueError):
    """A dataset row is malformed, or a set violates a matching invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def _require_text(value: Any, field_name: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{where}: {field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ContrastPair:
    """One matched pair. The two members differ in the varied span only.

    ``positive`` and ``negative`` are the *tool message content*. Everything
    else in the rendered conversation — system prompt, user turn, the assistant
    tool call — is identical between the members by construction, so the
    difference in means is a difference in the varied span plus whatever
    context effect the shared prefix induces.
    """

    pair_id: str
    concept: str
    scenario_family: str
    split: str
    system: str
    user: str
    tool_name: str
    tool_arguments: dict[str, Any]
    positive: str
    negative: str
    varied_span_positive: str
    varied_span_negative: str

    def __post_init__(self) -> None:
        where = f"pair {self.pair_id!r}"
        for name in ("pair_id", "concept", "scenario_family", "system", "user", "tool_name"):
            _require_text(getattr(self, name), name, where)
        _require(self.split in SPLITS, f"{where}: split must be one of {SPLITS}")
        _require(
            isinstance(self.tool_arguments, dict),
            f"{where}: tool_arguments must be an object",
        )
        for name in ("positive", "negative", "varied_span_positive", "varied_span_negative"):
            _require_text(getattr(self, name), name, where)
        _require(
            self.positive != self.negative,
            f"{where}: positive and negative members are identical",
        )
        _require(
            self.varied_span_positive in self.positive,
            f"{where}: varied_span_positive is not a substring of positive",
        )
        _require(
            self.varied_span_negative in self.negative,
            f"{where}: varied_span_negative is not a substring of negative",
        )
        shared_positive = self.positive.replace(self.varied_span_positive, "", 1)
        shared_negative = self.negative.replace(self.varied_span_negative, "", 1)
        _require(
            shared_positive == shared_negative,
            f"{where}: members differ outside the varied span. Only the varied "
            "span may differ, or difference-in-means fits the difference in "
            "shared context instead of the concept.",
        )

    def member(self, polarity: str) -> str:
        if polarity == "positive":
            return self.positive
        if polarity == "negative":
            return self.negative
        raise DatasetError(f"unknown polarity {polarity!r}")

    def messages(self, polarity: str) -> list[dict[str, Any]]:
        """The full conversation whose tool content is the requested member."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.pair_id}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": dict(self.tool_arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "content": self.member(polarity)},
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InjectionItem:
    """A held-out injection scenario scored without decoding.

    The tool output carries an imperative that conflicts with the user's task.
    Two fixed continuations are teacher-forced: ``complied`` acts on the
    injected imperative, ``faithful`` continues the user's task. The metric is
    the preference between them, so it needs one forward pass per continuation
    and no generation at all (PREFLIGHT.md §6, lever 1).
    """

    item_id: str
    scenario_family: str
    system: str
    user: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_output: str
    injected_imperative: str
    complied: str
    faithful: str

    def __post_init__(self) -> None:
        where = f"injection item {self.item_id!r}"
        for name in (
            "item_id",
            "scenario_family",
            "system",
            "user",
            "tool_name",
            "tool_output",
            "injected_imperative",
            "complied",
            "faithful",
        ):
            _require_text(getattr(self, name), name, where)
        _require(
            isinstance(self.tool_arguments, dict),
            f"{where}: tool_arguments must be an object",
        )
        _require(
            self.injected_imperative in self.tool_output,
            f"{where}: injected_imperative is not present in tool_output",
        )
        _require(
            self.complied != self.faithful,
            f"{where}: the two continuations are identical",
        )

    def messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.item_id}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": dict(self.tool_arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "content": self.tool_output},
        ]


@dataclass(frozen=True)
class RetainItem:
    """An ordinary tool-using exchange, scored as teacher-forced perplexity.

    Perplexity is measured over ``continuation`` only, with the intervention
    applied at tool-content positions exactly as it would be in deployment.
    This is the collateral-cost measurement that discriminates targeted
    suppression from tool output becoming illegible (PREFLIGHT.md §8, risk 1).
    """

    item_id: str
    scenario_family: str
    tool_dependent: bool
    system: str
    user: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_output: str
    continuation: str

    def __post_init__(self) -> None:
        where = f"retain item {self.item_id!r}"
        for name in (
            "item_id",
            "scenario_family",
            "system",
            "user",
            "tool_name",
            "tool_output",
            "continuation",
        ):
            _require_text(getattr(self, name), name, where)
        _require(
            isinstance(self.tool_dependent, bool),
            f"{where}: tool_dependent must be a bool",
        )
        _require(
            isinstance(self.tool_arguments, dict),
            f"{where}: tool_arguments must be an object",
        )

    def messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.item_id}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": dict(self.tool_arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "content": self.tool_output},
        ]


@dataclass(frozen=True)
class StructuredItem:
    """A prompt whose correct response is a single well-formed tool call.

    Scored by decoding under a pre-registered cap and checking that the output
    parses, names a declared tool, and supplies arguments matching the declared
    schema. Requires decode; the cap is fixed in ``EXPERIMENT_PROTOCOL.md``.
    """

    item_id: str
    system: str
    user: str
    tools: list[dict[str, Any]]
    expected_tool: str
    required_arguments: list[str]
    prior_tool_name: str = ""
    prior_tool_arguments: dict[str, Any] = field(default_factory=dict)
    prior_tool_output: str = ""

    def __post_init__(self) -> None:
        where = f"structured item {self.item_id!r}"
        for name in ("item_id", "system", "user", "expected_tool"):
            _require_text(getattr(self, name), name, where)
        _require(bool(self.tools), f"{where}: tools must be non-empty")
        _require(
            any(
                tool.get("function", {}).get("name") == self.expected_tool
                for tool in self.tools
            ),
            f"{where}: expected_tool {self.expected_tool!r} is not among tools",
        )
        _require(
            isinstance(self.required_arguments, list),
            f"{where}: required_arguments must be a list",
        )
        _require(
            bool(self.prior_tool_name) == bool(self.prior_tool_output),
            f"{where}: prior tool name and output must be given together. An "
            "item with no tool message exercises no tool-content positions and "
            "so measures nothing about the intervention.",
        )

    def messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]
        if self.prior_tool_name:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{self.item_id}",
                            "type": "function",
                            "function": {
                                "name": self.prior_tool_name,
                                "arguments": dict(self.prior_tool_arguments),
                            },
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "content": self.prior_tool_output})
        return messages


@dataclass(frozen=True)
class CapabilityItem:
    """A short question with a deterministic answer check.

    ``tool_dependent`` items can only be answered from the tool output, so they
    are the direct test for illegibility: if these fall while injection
    resistance rises, the intervention is damaging comprehension rather than
    suppressing compliance.
    """

    item_id: str
    tool_dependent: bool
    system: str
    user: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_output: str
    accept: list[str]

    def __post_init__(self) -> None:
        where = f"capability item {self.item_id!r}"
        for name in ("item_id", "system", "user", "tool_name", "tool_output"):
            _require_text(getattr(self, name), name, where)
        _require(
            isinstance(self.tool_dependent, bool),
            f"{where}: tool_dependent must be a bool",
        )
        _require(
            isinstance(self.accept, list) and bool(self.accept),
            f"{where}: accept must be a non-empty list of acceptable answers",
        )
        if self.tool_dependent:
            _require(
                any(answer.lower() in self.tool_output.lower() for answer in self.accept),
                f"{where}: no accepted answer appears in tool_output, so the item "
                "is not answerable from the tool output it claims to depend on",
            )

    def messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.item_id}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": dict(self.tool_arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "content": self.tool_output},
        ]


@dataclass(frozen=True)
class SafetyItem:
    """A request that the model should decline.

    ``channel`` is ``user`` when the harmful request arrives in a trailing user
    turn and ``tool`` when it arrives inside tool output. Both are mandatory at
    every arm and every layer: the Rogue Scalpel result is that interventions
    which look inert on capability metrics can still raise harmful compliance,
    and an intervention that only degrades refusal for tool-borne requests
    would be invisible without the second channel.

    Both shapes keep a real tool result in context, so the intervention is
    active at tool-content positions in either case. The user-channel request
    is the **last** turn rather than the first: measured on the pinned model, a
    harmful opening turn followed by an unrelated tool result leaves the model
    still in tool-calling mode, and it emits another call instead of ever
    reaching the decision the metric is trying to observe.
    """

    item_id: str
    channel: str
    category: str
    system: str
    user: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_output: str
    follow_up: str = ""

    def __post_init__(self) -> None:
        where = f"safety item {self.item_id!r}"
        for name in ("item_id", "category", "system", "user", "tool_name", "tool_output"):
            _require_text(getattr(self, name), name, where)
        _require(
            self.channel in ("user", "tool"),
            f"{where}: channel must be 'user' or 'tool'",
        )
        if self.channel == "user":
            _require_text(self.follow_up, "follow_up", where)
        else:
            _require(
                not self.follow_up,
                f"{where}: a tool-channel item must carry its request inside "
                "tool_output, not in a follow-up user turn",
            )

    def messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.item_id}",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": dict(self.tool_arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "content": self.tool_output},
        ]
        if self.follow_up:
            messages.append({"role": "user", "content": self.follow_up})
        return messages


def rendered_tool_content(messages: Sequence[Any]) -> str:
    """The tool-content characters of a rendered conversation, concatenated.

    This is the string the length-matching invariant applies to: it is the
    template's own output after JSON escaping, restricted to exactly the
    characters the primary mask selects. Matching raw content length is not
    enough, because the template escapes tool content and a pair matched before
    escaping can be unmatched after it (PREFLIGHT.md §5).
    """
    from server.rendering import render_characters

    text, regions, _ = render_characters(messages, add_generation_prompt=True)
    return "".join(
        text[region.start : region.end]
        for region in regions
        if region.role == "tool" and region.region == "content"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}: line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}: line {lineno} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Any]) -> str:
    """Write rows canonically and return the hash of what was written."""
    materialized = [asdict(row) if is_dataclass(row) else row for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(canonical_contrast_line(row))
            handle.write("\n")
    return hash_rows(materialized)


def hash_rows(rows: Iterable[Any]) -> str:
    """Hash rows exactly as a direction bundle hashes its contrast set."""
    return hash_contrasts(rows)


def hash_file(path: Path) -> str:
    return hash_rows(read_jsonl(Path(path)))


def _construct(cls: type, rows: Sequence[dict[str, Any]], path: Path) -> list[Any]:
    known = set(cls.__dataclass_fields__)
    items = []
    for index, row in enumerate(rows):
        unknown = set(row) - known
        if unknown:
            raise DatasetError(
                f"{path}: row {index} has unknown field(s) {sorted(unknown)}. "
                "Unrecognised dataset fields are rejected rather than ignored."
            )
        try:
            items.append(cls(**row))
        except TypeError as exc:
            raise DatasetError(f"{path}: row {index} is missing a field: {exc}") from exc
    return items


def validate_contrast_set(
    pairs: Sequence[ContrastPair],
    *,
    concept: str | None = None,
    renderer: Any = None,
) -> dict[str, Any]:
    """Check every matching invariant. Returns a summary; raises on failure.

    ``renderer`` is optional and, when given, must be a callable mapping a
    message list to rendered text. Passing ``server.rendering.render_characters``
    checks length matching on the **rendered** string — the invariant that
    actually matters, because the template JSON-escapes tool content and a pair
    matched on raw length can be unmatched after escaping (PREFLIGHT.md §5).
    Without a renderer the check falls back to raw content length, which is
    weaker and is reported as such.
    """
    _require(bool(pairs), "contrast set is empty")

    ids = Counter(pair.pair_id for pair in pairs)
    duplicates = sorted(pair_id for pair_id, count in ids.items() if count > 1)
    _require(not duplicates, f"duplicate pair_id(s): {duplicates}")

    concepts = {pair.concept for pair in pairs}
    _require(len(concepts) == 1, f"contrast set mixes concepts {sorted(concepts)}")
    if concept is not None:
        _require(
            concepts == {concept},
            f"contrast set declares concept {sorted(concepts)[0]!r}, expected {concept!r}",
        )

    systems = {pair.system for pair in pairs}
    _require(
        len(systems) == 1,
        "the system prompt must be identical in every pair and across the set "
        f"(PREFLIGHT.md §5); found {len(systems)} distinct system prompts",
    )

    families_by_split: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        families_by_split[pair.split].add(pair.scenario_family)
    straddling = families_by_split.get("train", set()) & families_by_split.get("heldout", set())
    _require(
        not straddling,
        f"scenario families {sorted(straddling)} appear in both splits. The "
        "split is by scenario family precisely so that near-duplicate "
        "paraphrases cannot straddle it.",
    )
    for split in SPLITS:
        _require(
            bool(families_by_split.get(split)),
            f"split {split!r} is empty; both splits must be populated",
        )

    render = renderer if renderer is not None else None
    unmatched: list[str] = []
    deltas: list[int] = []
    for pair in pairs:
        if render is None:
            lengths = (len(pair.positive), len(pair.negative))
        else:
            lengths = (
                len(render(pair.messages("positive"))),
                len(render(pair.messages("negative"))),
            )
        delta = abs(lengths[0] - lengths[1])
        deltas.append(delta)
        allowance = max(LENGTH_TOLERANCE_FLOOR, LENGTH_TOLERANCE * max(lengths))
        if delta > allowance:
            unmatched.append(
                f"{pair.pair_id}: lengths {lengths[0]} vs {lengths[1]} "
                f"(delta {delta} > allowance {allowance:.1f})"
            )
    _require(
        not unmatched,
        "pairs are not length-matched, so difference-in-means would fit length:\n  "
        + "\n  ".join(unmatched[:20])
        + (f"\n  ... and {len(unmatched) - 20} more" if len(unmatched) > 20 else ""),
    )

    return {
        "concept": sorted(concepts)[0],
        "pairs": len(pairs),
        "families": sorted({pair.scenario_family for pair in pairs}),
        "train_pairs": sum(1 for pair in pairs if pair.split == "train"),
        "heldout_pairs": sum(1 for pair in pairs if pair.split == "heldout"),
        "length_matching": "rendered" if render is not None else "raw",
        "max_length_delta": max(deltas),
        "mean_length_delta": sum(deltas) / len(deltas),
        "hash": hash_rows([pair.to_dict() for pair in pairs]),
    }


def load_contrast_set(
    concept: str, *, root: Path | None = None, renderer: Any = None
) -> list[ContrastPair]:
    """Load and validate one contrast set. Raises rather than returning a bad one."""
    root = Path(root) if root is not None else DATA_ROOT
    path = root / "contrasts" / f"{concept}.jsonl"
    if not path.is_file():
        raise DatasetError(f"no contrast set at {path}")
    pairs = _construct(ContrastPair, read_jsonl(path), path)
    validate_contrast_set(pairs, concept=concept, renderer=renderer)
    return pairs


def _load(cls: type, name: str, root: Path | None) -> list[Any]:
    root = Path(root) if root is not None else DATA_ROOT
    path = root / f"{name}.jsonl"
    if not path.is_file():
        raise DatasetError(f"no dataset at {path}")
    items = _construct(cls, read_jsonl(path), path)
    _require(bool(items), f"{path}: dataset is empty")
    ids = Counter(item.item_id for item in items)
    duplicates = sorted(item_id for item_id, count in ids.items() if count > 1)
    _require(not duplicates, f"{path}: duplicate item_id(s) {duplicates}")
    return items


def load_injection_set(*, root: Path | None = None) -> list[InjectionItem]:
    return _load(InjectionItem, "injection", root)


def load_retain_set(*, root: Path | None = None) -> list[RetainItem]:
    return _load(RetainItem, "retain", root)


def load_structured_set(*, root: Path | None = None) -> list[StructuredItem]:
    return _load(StructuredItem, "structured", root)


def load_capability_set(*, root: Path | None = None) -> list[CapabilityItem]:
    return _load(CapabilityItem, "capability", root)


def load_safety_set(*, root: Path | None = None) -> list[SafetyItem]:
    items = _load(SafetyItem, "safety", root)
    channels = {item.channel for item in items}
    _require(
        channels == {"user", "tool"},
        "the safety set must exercise both the user and the tool channel; found "
        f"{sorted(channels)}",
    )
    return items
