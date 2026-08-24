"""Diagnostic CLI for rendered character and token provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .rendering import load_pinned_tokenizer, render_chat


def _request(path: str) -> dict[str, Any]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError("input must be a JSON object containing a messages list")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="chat request JSON path, or - for stdin")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    request = _request(args.request)
    tokenizer = load_pinned_tokenizer(local_files_only=args.local_files_only)
    rendered = render_chat(
        tokenizer, request["messages"], tools=request.get("tools"), add_generation_prompt=True
    )

    print("RENDERED TEXT")
    print(rendered.text)
    print("\nCHARACTER REGIONS")
    for region in rendered.regions:
        snippet = rendered.text[region.start : region.end]
        print(
            f"{region.start:6}:{region.end:<6} role={region.role!r:<12} "
            f"region={region.region:<18} message={region.message_index!r:<4} {snippet!r}"
        )
    ambiguous = {item.token_index for item in rendered.ambiguous_tokens}
    print("\nTOKENS")
    for index, (token_id, offsets) in enumerate(zip(rendered.input_ids, rendered.offsets)):
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print(
            f"{index:5} id={token_id:<7} span={offsets!s:<14} token={decoded!r:<24} "
            f"tool_content={int(rendered.primary_mask[index])} "
            f"whole_tool_block={int(rendered.whole_tool_block_mask[index])} "
            f"ambiguous={int(index in ambiguous)}"
        )
    print("\nAMBIGUOUS TOKENS")
    for item in rendered.ambiguous_tokens:
        print(
            f"token={item.token_index} id={item.token_id} span=({item.start}, {item.end}) "
            f"regions={','.join(item.regions)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
