# Checkpoint 3 span mapping

The shim supports only the pinned Llama 3.1 Instruct template. It constructs
the rendered prompt and provenance together. It never searches the completed
text for role delimiters: tool output can contain valid Llama special-token
strings, so delimiter-based recovery would let untrusted content forge its own
provenance.

## Character regions

Every rendered character belongs to exactly one region:

- `content`: a character originating in a message;
- `role_marker`: the complete template-generated role header;
- `separator`: newlines, JSON string quotes, and JSON escape syntax;
- `special_token`: BOS, EOT, and other template control tokens;
- `tool_call_envelope`: template-generated tool instructions and serialized
  assistant calls.

For a string tool result, the official template applies JSON encoding. The
opening and closing quotes are separators. For an escaped source character,
the escape syntax is a separator and the final encoded character carries the
source provenance. Thus each source character has one owner even when its
rendered representation is longer. Mapping or list content is serialized as a
JSON value and the entire serialization is tool content.

## Token boundary rule

The primary mask is selected when a token's offset has **any overlap** with a
character whose provenance is `role == "tool"` and `region == "content"`.
This inclusive rule avoids leaving an input character untreated merely because
the tokenizer merged it with a template quote or escape. It also means that a
mixed token unavoidably carries a small amount of template context through the
hook. Every token overlapping more than one provenance class is retained in
`RenderedChat.ambiguous_tokens` and printed by `inspect-spans`; the spillover is
observable rather than silently classified as pure content.

The secondary `whole_tool_block` mask uses the same any-overlap rule from the
start of the `ipython` header through its terminating EOT. It is stored
separately and is selected only by an explicit top-level intervention config.

The trusted request field is separate from `messages`:

```json
{
  "intervention": {
    "enabled": true,
    "direction_id": "compliance-v1",
    "layer": 19,
    "mode": "project",
    "scope": "tool_content",
    "direction_norm": "unit",
    "alpha": 0.0
  }
}
```

`mode` is `project` or the comparison arm `add`; `alpha` is used only by
`add`. `direction_norm` makes unit-versus-raw handling explicit. A raw
projection direction is normalized before projection, while raw additive
steering preserves its stored norm. Content inside any message is data and
cannot alter this top-level state.

Empty string tool output renders as `""`. Both quotes are template separators,
so its primary mask is correctly empty while its whole-block mask is not.

## Runtime position rule

The hook maps full-prefix prefill positions to the selected prompt mask. When
Transformers supplies `cache_position` or Llama's `position_ids`, those
absolute positions take precedence. Incremental positions at or beyond the
prompt length are newly generated assistant tokens and are never selected. In
a no-cache full-prefix forward, positions beyond the prompt length likewise
map to false.

Requests are serialized by one async lock. The active state uses a context
variable, is installed only inside that lock, and is reset in `finally`. The
model hook is also removed in `finally`, including when generation raises.

## Diagnostic command

Pass an OpenAI-style request object containing `messages` and optional `tools`:

```console
uv run inspect-spans --local-files-only request.json
```

The command prints the exact rendered text, character regions, token IDs,
decoded token text, both masks, and all ambiguous tokens. It is diagnostic
output only and does not modify the experiment.
