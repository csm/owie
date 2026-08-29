"""Llama 3 chat rendering with character and token provenance.

The implementation intentionally mirrors the pinned tokenizer template rather
than trying to recover provenance from delimiters after rendering. Tool output
is allowed to contain those delimiters, so delimiter recognition is unsound.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PILOT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
PILOT_MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
CHAT_TEMPLATE_DATE = "26 Jul 2024"

_LOGGER = logging.getLogger(__name__)

BOS = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"


class RenderError(ValueError):
    """A request cannot be represented by the pinned chat template."""


@dataclass(frozen=True)
class CharacterRegion:
    start: int
    end: int
    role: str | None
    region: str
    message_index: int | None = None
    source_character: int | None = None


@dataclass(frozen=True)
class AmbiguousToken:
    token_index: int
    token_id: int
    start: int
    end: int
    regions: tuple[str, ...]


@dataclass(frozen=True)
class RenderedChat:
    text: str
    regions: tuple[CharacterRegion, ...]
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    primary_mask: tuple[bool, ...]
    whole_tool_block_mask: tuple[bool, ...]
    ambiguous_tokens: tuple[AmbiguousToken, ...]


class _Writer:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.regions: list[CharacterRegion] = []
        self.tool_blocks: list[tuple[int, int]] = []
        self.length = 0

    def append(
        self,
        text: str,
        region: str,
        *,
        role: str | None = None,
        message_index: int | None = None,
        source_character: int | None = None,
    ) -> None:
        if not text:
            return
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        self.regions.append(
            CharacterRegion(
                start,
                self.length,
                role,
                region,
                message_index,
                source_character,
            )
        )

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json(value: Any, *, indent: int | None = None) -> str:
    # Matches the Transformers 5.15.1 Jinja environment: non-ASCII and mapping
    # insertion order are retained by its tojson policy.
    return json.dumps(value, ensure_ascii=False, sort_keys=False, indent=indent)


def _append_header(writer: _Writer, role: str) -> None:
    writer.append(f"{START_HEADER}{role}{END_HEADER}", "role_marker", role=role)
    writer.append("\n\n", "separator", role=role)


def _append_special(writer: _Writer, text: str, role: str | None = None) -> None:
    writer.append(text, "special_token", role=role)


def _trimmed_content(content: Any) -> tuple[str, int]:
    if not isinstance(content, str):
        raise RenderError("system, user, and ordinary assistant content must be strings")
    trimmed = content.strip()
    if not trimmed:
        return "", len(content)
    return trimmed, content.index(trimmed)


def _append_raw_content(
    writer: _Writer, content: str, role: str, message_index: int, source_offset: int
) -> None:
    for offset, char in enumerate(content):
        writer.append(
            char,
            "content",
            role=role,
            message_index=message_index,
            source_character=source_offset + offset,
        )


def _append_json_string_content(
    writer: _Writer, content: str, role: str, message_index: int
) -> None:
    writer.append('"', "separator", role=role, message_index=message_index)
    for source_index, char in enumerate(content):
        encoded = _json(char)[1:-1]
        # JSON escape syntax is template-generated. The final rendered
        # character represents the source character and retains its provenance.
        if len(encoded) > 1:
            writer.append(
                encoded[:-1], "separator", role=role, message_index=message_index
            )
        writer.append(
            encoded[-1],
            "content",
            role=role,
            message_index=message_index,
            source_character=source_index,
        )
    writer.append('"', "separator", role=role, message_index=message_index)


def _tool_function(tool_call: Any) -> Any:
    function = _value(tool_call, "function")
    if function is None:
        raise RenderError("tool_calls entries require a function object")
    return function


def _normalize_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Validate OpenAI roles and decode wire-format tool arguments."""
    normalized: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise RenderError(f"message {message_index} must be an object")
        item = dict(message)
        role = item.get("role")
        if role == "ipython":
            raise RenderError(
                "role 'ipython' is not accepted on the OpenAI-compatible surface; "
                "send tool results with role 'tool'"
            )

        tool_calls = item.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant":
                raise RenderError(
                    f"message {message_index} has tool_calls but role is {role!r}; "
                    "tool_calls are valid only on assistant messages"
                )
            if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str):
                raise RenderError("tool_calls must be a list")
            normalized_calls: list[dict[str, Any]] = []
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, Mapping):
                    raise RenderError(f"tool_calls[{call_index}] must be an object")
                call = dict(tool_call)
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise RenderError(
                        f"tool_calls[{call_index}].function must be an object"
                    )
                function_copy = dict(function)
                arguments = function_copy.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise RenderError(
                            f"tool_calls[{call_index}].function.arguments is not valid JSON"
                        ) from exc
                if not isinstance(arguments, Mapping):
                    raise RenderError(
                        f"tool_calls[{call_index}].function.arguments must encode a JSON object"
                    )
                function_copy["arguments"] = dict(arguments)
                call["function"] = function_copy
                normalized_calls.append(call)
            item["tool_calls"] = normalized_calls
        normalized.append(item)
    return normalized


def _render_tools_preamble(
    writer: _Writer, tools: Sequence[Any], first_user: tuple[int, Any]
) -> None:
    index, message = first_user
    if _value(message, "role") != "user":
        raise RenderError(
            "the pinned template requires the first remaining message to be user"
        )
    content, source_offset = _trimmed_content(_value(message, "content"))
    _append_header(writer, "user")
    writer.append(
        "Given the following functions, please respond with a JSON for a function call "
        "with its proper arguments that best answers the given prompt.\n\n"
        'Respond in the format {"name": function name, "parameters": dictionary '
        "of argument name and its value}."
        "Do not use variables.\n\n",
        "tool_call_envelope",
        role="user",
    )
    for tool in tools:
        writer.append(_json(tool, indent=4), "tool_call_envelope", role="user")
        writer.append("\n\n", "separator", role="user")
    _append_raw_content(writer, content, "user", index, source_offset)
    _append_special(writer, EOT, "user")


def _render_characters_normalized(
    messages: Sequence[Any],
    *,
    tools: Sequence[Any] | None = None,
    add_generation_prompt: bool = True,
) -> tuple[str, tuple[CharacterRegion, ...], tuple[tuple[int, int], ...]]:
    """Render the pinned Llama template and retain provenance as it is emitted."""
    if not messages:
        raise RenderError("messages must contain at least one item")
    writer = _Writer()
    _append_special(writer, BOS)

    indexed = list(enumerate(messages))
    system_message = ""
    system_index: int | None = None
    system_source_offset = 0
    if _value(indexed[0][1], "role") == "system":
        system_index, system = indexed.pop(0)
        system_message, system_source_offset = _trimmed_content(_value(system, "content"))

    _append_header(writer, "system")
    if tools is not None:
        writer.append("Environment: ipython\n", "separator", role="system")
    writer.append(
        f"Cutting Knowledge Date: December 2023\nToday Date: {CHAT_TEMPLATE_DATE}\n\n",
        "separator",
        role="system",
    )
    if system_index is not None:
        _append_raw_content(
            writer, system_message, "system", system_index, system_source_offset
        )
    _append_special(writer, EOT, "system")

    if tools is not None:
        if not indexed:
            raise RenderError("cannot place tools in a user message without a user message")
        _render_tools_preamble(writer, tools, indexed.pop(0))

    for message_index, message in indexed:
        role = _value(message, "role")
        tool_calls = _value(message, "tool_calls", None)
        if tool_calls is not None:
            if len(tool_calls) != 1:
                raise RenderError(
                    "the pinned model supports exactly one tool call per message"
                )
            function = _tool_function(tool_calls[0])
            name = _value(function, "name")
            arguments = _value(function, "arguments")
            if not isinstance(name, str):
                raise RenderError("tool call function name must be a string")
            _append_header(writer, "assistant")
            writer.append(
                '{"name": "' + name + '", "parameters": ' + _json(arguments) + "}",
                "tool_call_envelope",
                role="assistant",
                message_index=message_index,
            )
            _append_special(writer, EOT, "assistant")
            continue

        if role in {"tool", "ipython"}:
            block_start = writer.length
            _append_header(writer, "ipython")
            content = _value(message, "content")
            if isinstance(content, str):
                _append_json_string_content(writer, content, role, message_index)
            elif isinstance(content, (Mapping, list, tuple)):
                writer.append(
                    _json(content),
                    "content",
                    role=role,
                    message_index=message_index,
                )
            else:
                writer.append(
                    str(content),
                    "content",
                    role=role,
                    message_index=message_index,
                )
            _append_special(writer, EOT, role)
            writer.tool_blocks.append((block_start, writer.length))
            continue

        if not isinstance(role, str) or not role:
            raise RenderError("every message requires a non-empty role")
        content, source_offset = _trimmed_content(_value(message, "content"))
        _append_header(writer, role)
        _append_raw_content(writer, content, role, message_index, source_offset)
        _append_special(writer, EOT, role)

    if add_generation_prompt:
        _append_header(writer, "assistant")
    return writer.text, tuple(writer.regions), tuple(writer.tool_blocks)


def render_characters(
    messages: Sequence[Any],
    *,
    tools: Sequence[Any] | None = None,
    add_generation_prompt: bool = True,
) -> tuple[str, tuple[CharacterRegion, ...], tuple[tuple[int, int], ...]]:
    """Render after validating and normalizing OpenAI wire-format messages."""
    return _render_characters_normalized(
        _normalize_messages(messages),
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )


def _overlaps(start: int, end: int, region_start: int, region_end: int) -> bool:
    return start < region_end and region_start < end


def _token_regions(
    regions: tuple[CharacterRegion, ...],
    starts: tuple[int, ...],
    start: int,
    end: int,
) -> list[CharacterRegion]:
    """Return overlapping regions without rescanning the complete prompt."""
    if start >= end or not regions:
        return []
    index = max(0, bisect_right(starts, start) - 1)
    overlapping: list[CharacterRegion] = []
    while index < len(regions) and regions[index].start < end:
        region = regions[index]
        if _overlaps(start, end, region.start, region.end):
            overlapping.append(region)
        index += 1
    return overlapping


def render_chat(
    tokenizer: Any,
    messages: Sequence[Any],
    *,
    tools: Sequence[Any] | None = None,
    add_generation_prompt: bool = True,
    verify_official: bool = True,
) -> RenderedChat:
    normalized_messages = _normalize_messages(messages)
    text, regions, tool_blocks = _render_characters_normalized(
        normalized_messages, tools=tools, add_generation_prompt=add_generation_prompt
    )
    if verify_official:
        official = tokenizer.apply_chat_template(
            normalized_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            date_string=CHAT_TEMPLATE_DATE,
        )
        if text != official:
            raise RenderError(
                "provenance renderer differs from the pinned tokenizer's official chat template"
            )

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = tuple((int(start), int(end)) for start, end in encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise RenderError("tokenizer returned different token and offset counts")

    primary: list[bool] = []
    whole: list[bool] = []
    ambiguous: list[AmbiguousToken] = []
    region_starts = tuple(region.start for region in regions)
    for token_index, (token_id, (start, end)) in enumerate(zip(input_ids, offsets)):
        overlapping = _token_regions(regions, region_starts, start, end)
        primary.append(
            any(r.role == "tool" and r.region == "content" for r in overlapping)
        )
        whole.append(any(_overlaps(start, end, a, b) for a, b in tool_blocks))
        labels = tuple(sorted({f"{r.role or '-'}:{r.region}" for r in overlapping}))
        if len(labels) > 1:
            item = AmbiguousToken(token_index, token_id, start, end, labels)
            ambiguous.append(item)
            _LOGGER.debug(
                "ambiguous token index=%d id=%d offsets=(%d,%d) regions=%s",
                token_index,
                token_id,
                start,
                end,
                ",".join(labels),
            )

    return RenderedChat(
        text,
        regions,
        input_ids,
        offsets,
        tuple(primary),
        tuple(whole),
        tuple(ambiguous),
    )


def load_pinned_tokenizer(*, local_files_only: bool = False) -> Any:
    """Load only the immutable tokenizer revision used by the experiment."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only
    )


def load_pilot_tokenizer(*, local_files_only: bool = False) -> Any:
    """Load the approved non-reporting 3B pilot at its immutable revision."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        PILOT_MODEL_ID,
        revision=PILOT_MODEL_REVISION,
        local_files_only=local_files_only,
    )
