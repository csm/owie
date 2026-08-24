"""Load the pinned model and run the one-worker experimental server."""

from __future__ import annotations

import argparse
from typing import Sequence

from .api import create_app
from .backend import RegisteredDirection, TransformersBackend, hash_direction_bundle
from .rendering import MODEL_ID, MODEL_REVISION, load_pinned_tokenizer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="DIRECTION_ID",
        help="load and verify a directions/<id> bundle; may be repeated",
    )
    args = parser.parse_args(argv)

    import uvicorn
    from directions import read_bundle
    from transformers import AutoModelForCausalLM

    tokenizer = load_pinned_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype="auto", device_map="mps"
    )
    directions = {}
    for direction_id in args.direction:
        bundle = read_bundle(direction_id)
        if (
            bundle.manifest.model_id != MODEL_ID
            or bundle.manifest.model_revision != MODEL_REVISION
        ):
            parser.error(
                f"direction {direction_id!r} targets "
                f"{bundle.manifest.model_id}@{bundle.manifest.model_revision}, "
                "not the pinned model"
            )
        directions[direction_id] = RegisteredDirection(
            bundle.vector,
            layer=bundle.manifest.layer,
            normalization=bundle.manifest.normalization,
            hook_point=bundle.manifest.hook_point,
            bundle_hash=hash_direction_bundle(bundle.path),
        )
    app = create_app(
        TransformersBackend(model, tokenizer, directions=directions), tokenizer
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
