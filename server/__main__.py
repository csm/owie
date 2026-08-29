"""Load the pinned model and run the one-worker experimental server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .api import create_app
from .backend import (
    RegisteredDirection,
    RegisteredSAEFeature,
    TransformersBackend,
    hash_direction_bundle,
    registered_sham,
)
from .rendering import (
    MODEL_ID,
    MODEL_REVISION,
    PILOT_MODEL_ID,
    PILOT_MODEL_REVISION,
    load_pilot_tokenizer,
    load_pinned_tokenizer,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--pilot-3b",
        action="store_true",
        help="serve the approved non-reporting 3B determinism pilot",
    )
    parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="DIRECTION_ID",
        help="load and verify a directions/<id> bundle; may be repeated",
    )
    parser.add_argument("--direction-root", type=Path, default=Path("directions"))
    parser.add_argument("--sham-seed", action="append", type=int, default=[])
    parser.add_argument("--sae-c1-rank0", action="store_true")
    parser.add_argument("--sae-selection", type=Path)
    parser.add_argument("--sae-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn
    from directions import read_bundle
    from transformers import AutoModelForCausalLM

    model_id = PILOT_MODEL_ID if args.pilot_3b else MODEL_ID
    model_revision = PILOT_MODEL_REVISION if args.pilot_3b else MODEL_REVISION
    tokenizer = (
        load_pilot_tokenizer(local_files_only=args.local_files_only)
        if args.pilot_3b
        else load_pinned_tokenizer(local_files_only=args.local_files_only)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        dtype="auto",
        device_map="mps",
        local_files_only=args.local_files_only,
    )
    directions = {}
    for direction_id in args.direction:
        bundle = read_bundle(direction_id, root=args.direction_root)
        if (
            bundle.manifest.model_id != model_id
            or bundle.manifest.model_revision != model_revision
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
    d_model = int(model.config.hidden_size)
    for seed in args.sham_seed:
        directions[f"sham-{seed}"] = registered_sham(seed, d_model, layer=10)

    sae_features = {}
    if args.sae_c1_rank0:
        if args.pilot_3b:
            parser.error("the frozen SAE arm is not available for the 3B pilot")
        if args.sae_selection is None or args.sae_cache_dir is None:
            parser.error("--sae-c1-rank0 requires --sae-selection and --sae-cache-dir")
        from directions.sae import load_sae
        from .runtime import SAEFeature

        selection = json.loads(args.sae_selection.read_text(encoding="utf-8"))["c1"]
        if selection.get("selection_hash") != (
            "sha256:2de8241291dc5504f71abb7926ca2e83f9a04ae9a8bc6618b31f3d6e1493eab2"
        ):
            parser.error("the C1 SAE selection hash does not match the frozen protocol")
        feature_index = int(selection["features"][0]["feature_index"])
        if feature_index != 1584:
            parser.error("the frozen C1 rank-zero SAE feature is not 1584")
        sae = load_sae(
            cache_dir=args.sae_cache_dir,
            local_files_only=args.local_files_only,
        )
        if sae.safetensors_hash != (
            "sha256:5223dd47c15704c036fef4cbec5feb45355e4b60db7676a4e4e80f1d62cec66d"
        ):
            parser.error("the SAE weights hash does not match the frozen protocol")
        encoder_row, encoder_bias, decoder_column = sae.feature(feature_index)
        artifact_digest = hashlib.sha256(
            f"{sae.safetensors_hash}\0{selection['selection_hash']}\0{feature_index}".encode()
        ).hexdigest()
        feature_id = "sae-c1-rank0-feature-1584"
        sae_features[feature_id] = RegisteredSAEFeature(
            SAEFeature(
                feature_index=feature_index,
                encoder_row=encoder_row,
                encoder_bias=encoder_bias,
                decoder_column=decoder_column,
                clamp_value=0.0,
                sae_hash=sae.safetensors_hash,
            ),
            layer=sae.layer,
            artifact_hash=f"sha256:{artifact_digest}",
        )
        del sae
    app = create_app(
        TransformersBackend(
            model,
            tokenizer,
            directions=directions,
            sae_features=sae_features,
        ),
        tokenizer,
        model_id=model_id,
        model_revision=model_revision,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
