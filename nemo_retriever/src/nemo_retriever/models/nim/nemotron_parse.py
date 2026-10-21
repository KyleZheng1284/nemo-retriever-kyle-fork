# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-team document parsing over an OpenAI-compatible served endpoint."""

from __future__ import annotations

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Sequence

from nemo_retriever.models.nim.nim import NIMClient, _parse_invoke_urls
from nemo_retriever.models.nim.parse_rendering import (
    attempt_record,
    eligible,
    image_record,
    prepare_retry,
)


def invoke_document_parse_images(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    model: str,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
    nim_client: NIMClient | None = None,
    mode: str = "single",
    triple_regions: str = "union",
    rendering_retry: bool = True,
    page_pdfs: Sequence[bytes | None] | None = None,
    page_identities: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Parse pages concurrently with the model package's prompts and sampling.

    Results retain input order. Each failed page has raw output, completion
    metadata, and an error; a page failure does not discard successful siblings.
    The optional package is imported only for this served model contract.
    """
    try:
        from nemotron_parse.inference import DocumentParser, GenerationError
    except ImportError as exc:
        raise ImportError(
            "Served Nemotron Parse 3.5 requires the optional nemotron-parse package. "
            "Install the model team's pinned package with --no-deps: "
            "pip install --no-deps 'git+https://huggingface.co/nvidia/Nemotron-3.5-Parse"
            "@e7ea79242a87ea0ad3919b672bfedb429a683206'. "
            "Refer to the Parse documentation for vLLM plugin isolation."
        ) from exc
    from PIL import Image

    if mode not in {"single", "double", "triple"} or triple_regions not in {"union", "detector"}:
        raise ValueError("Invalid Parse mode or triple_regions")
    page_pdfs = [None] * len(image_b64_list) if page_pdfs is None else page_pdfs
    page_identities = [{} for _ in image_b64_list] if page_identities is None else page_identities
    if len(page_pdfs) != len(image_b64_list) or len(page_identities) != len(image_b64_list):
        raise ValueError("Page PDF/identity count must match image count")
    endpoints = _parse_invoke_urls(invoke_url)
    client = nim_client or NIMClient(max_pool_workers=max_pool_workers)

    def parse_one(item: tuple[int, str]) -> dict[str, Any]:
        index, encoded = item
        endpoint = endpoints[index % len(endpoints)]
        attempt = 0
        records: list[dict[str, Any]] = []
        images = []
        result = {
            "raw_output": None,
            "routing_output": "",
            "finish_reason": None,
            "usage": None,
            "error": None,
            "attempts": records,
            "selected_attempt": 0,
            "rendering_retry": {"enabled": rendering_retry, "attempted": False, "selected": False},
        }

        def backend(image: Any, *, prompt: str, stage: str, settings: Any) -> dict[str, Any]:
            sampling = asdict(settings)
            record = attempt_record(
                image,
                attempt=attempt,
                stage=stage,
                prompt=prompt,
                settings={**sampling, "skip_special_tokens": False},
                identity=page_identities[index],
            )
            records.append(record)
            try:
                responses = client.invoke_chat_completions_images(
                    invoke_url=endpoint,
                    image_b64_list=[record["input_image"]["image_b64"]],
                    model=model,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    task_prompt=prompt,
                    temperature=sampling.pop("temperature"),
                    repetition_penalty=sampling.pop("repetition_penalty"),
                    extra_body={**sampling, "skip_special_tokens": False},
                    max_retries=max_retries,
                    max_429_retries=max_429_retries,
                    return_metadata=True,
                )
                if len(responses) != 1:
                    raise RuntimeError("Expected exactly one completion for the page request")
                response = responses[0]
                record.update(
                    raw_output=response["text"],
                    finish_reason=response.get("finish_reason"),
                    usage=response.get("usage"),
                )
                output = {
                    "raw_out": response["text"],
                    "finish_reason": response.get("finish_reason"),
                    "usage": response.get("usage"),
                }
                if attempt == 1 and stage == "fullpage" and response.get("finish_reason") != "stop":
                    output["error"] = "A rendering retry must finish with stop"
                return output
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                record["ended_unix"] = time.time()

        try:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as original:
                image = original.convert("RGB")
            image.info.clear()
            images.append(image)
            with DocumentParser(model=model, backend=backend) as parser:
                while True:
                    try:
                        document = parser.parse(image, mode=mode, triple_regions=triple_regions)
                        break
                    except GenerationError as exc:
                        if not (
                            rendering_retry and attempt == 0 and exc.stage == "fullpage" and eligible(exc.response)
                        ):
                            raise
                        candidate, reason = prepare_retry(image, page_pdfs[index], parse35=True)
                        result["rendering_retry"]["reason"] = reason
                        if candidate is None:
                            raise
                        image = candidate
                        images.append(image)
                        attempt = 1
                        result["rendering_retry"]["attempted"] = True
            first = document["stages"]["fullpage"]
            result.update(
                raw_output=first["raw_out"],
                routing_output=document["raw_out"],
                finish_reason=first.get("finish_reason"),
                usage=first.get("usage"),
                inference=document["inference"],
                stages=document["stages"],
                selected_attempt=attempt,
            )
            if attempt:
                result["rendering_retry"]["selected"] = True
                result["selected_page_image"] = image_record(image)
        except Exception as exc:
            fullpages = [r for r in records if r["stage"] == "fullpage"]
            if fullpages:
                # An unsuccessful retry cannot replace the original output. A crop failure
                # after a complete second full-page response still retains that response.
                selected = (
                    fullpages[-1]
                    if (
                        fullpages[-1]["finish_reason"] == "stop"
                        and str(fullpages[-1]["raw_output"] or "").strip()
                        and not fullpages[-1]["error"]
                    )
                    else fullpages[0]
                )
                result.update(
                    raw_output=selected["raw_output"],
                    finish_reason=selected["finish_reason"],
                    usage=selected["usage"],
                    selected_attempt=selected["attempt"],
                )
                result["rendering_retry"]["selected"] = bool(selected["attempt"])
                if selected["attempt"]:
                    result["selected_page_image"] = selected["input_image"]
            result["error"] = {
                "stage": "nemotron_parse_pages_" + str(getattr(exc, "stage", "request")),
                "type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            for image in images:
                image.close()
        return result

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_pool_workers))) as pool:
            return list(pool.map(parse_one, enumerate(image_b64_list)))
    finally:
        if nim_client is None:
            client.shutdown()
