# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One changed rendering, with input identity retained for every Parse attempt."""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
import uuid
from contextlib import ExitStack
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from nemo_retriever.common.api.util.pdf.process_lock import PDFIUM_PROCESS_LOCK

PAGE_PDF_COLUMN = "_nemotron_parse_page_pdf"
PARSE35_REVISION = "e7ea79242a87ea0ad3919b672bfedb429a683206"
PARSE_INPUT_WH = (1664, 2048)
_PROCESSOR_LOCK = threading.Lock()


def is_parse35(model: str | None) -> bool:
    return str(model or "").lower() in {"nvidia/nemotron-3.5-parse", "nemotron-3.5-parse"}


def retry_enabled(setting: bool | None, model: str | None, invoke_url: str | None = None) -> bool:
    if setting is not None:
        return setting
    urls = [url.strip() for url in str(invoke_url or "").split(",") if url.strip()]
    return (
        is_parse35(model)
        and bool(urls)
        and all(urlsplit(url).path.rstrip("/") == "/v1/chat/completions" for url in urls)
    )


def encode_image(image: Image.Image) -> bytes:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def image_record(image: Image.Image) -> dict[str, Any]:
    payload = encode_image(image)
    return {
        "image_b64": base64.b64encode(payload).decode("ascii"),
        "encoding": "png",
        "orig_shape_hw": (image.height, image.width),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def attempt_record(
    image: Image.Image,
    *,
    attempt: int,
    stage: str,
    prompt: str,
    settings: dict[str, Any] | None,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """The attempt ID is a client trace ID, not a provider-issued request ID."""
    return {
        "attempt_id": uuid.uuid4().hex,
        "attempt": attempt,
        "stage": stage,
        "page_identity": identity,
        "input_image": image_record(image),
        "prompt": prompt,
        "settings": settings,
        "started_unix": time.time(),
        "ended_unix": None,
        "raw_output": None,
        "finish_reason": None,
        "usage": None,
        "error": None,
    }


def eligible(response: dict[str, Any]) -> bool:
    return not response.get("error") and (
        response.get("finish_reason") == "length"
        or (response.get("finish_reason") == "stop" and not str(response.get("raw_out") or "").strip())
    )


@lru_cache(maxsize=1)
def _load_parse35_processor():
    from transformers import AutoImageProcessor

    from nemo_retriever.models.hf_cache import configure_global_hf_cache_base

    configure_global_hf_cache_base()
    return AutoImageProcessor.from_pretrained(
        "nvidia/Nemotron-3.5-Parse", revision=PARSE35_REVISION, trust_remote_code=True
    )


def effective_image_fingerprint(image: Image.Image, processor=None) -> str:
    """Use the pinned model processor, including its actual packing geometry."""
    with _PROCESSOR_LOCK:
        processor = processor or _load_parse35_processor()
        result = processor(images=image, return_tensors="np")
    digest = hashlib.sha256()
    for key in sorted(result):
        value = np.asarray(result[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def render_retry_image(page_pdf: bytes, target_wh: tuple[int, int] = PARSE_INPUT_WH) -> Image.Image:
    """Render one physical page and own the pixels before closing PDFium."""
    import pypdfium2 as pdfium

    from nemo_retriever.common.api.util.pdf.pdfium import (
        convert_bitmap_to_corrected_numpy,
    )

    with PDFIUM_PROCESS_LOCK, ExitStack() as stack:
        document = pdfium.PdfDocument(bytes(page_pdf))
        stack.callback(document.close)
        if len(document) != 1:
            raise ValueError("Rendering retry requires a single-page PDF")
        page = document[0]
        stack.callback(page.close)
        width, height = float(page.get_width()), float(page.get_height())
        if width <= 0 or height <= 0:
            raise ValueError("Rendering retry requires positive page dimensions")
        bitmap = page.render(scale=min(target_wh[0] / width, target_wh[1] / height))
        stack.callback(bitmap.close)
        image = Image.fromarray(convert_bitmap_to_corrected_numpy(bitmap).copy()).convert("RGB")
        image.info.clear()
        return image


def prepare_retry(
    image: Image.Image, page_pdf: bytes | None, *, parse35: bool, processor=None
) -> tuple[Image.Image | None, str]:
    if not isinstance(page_pdf, (bytes, bytearray, memoryview)) or not page_pdf:
        return None, "no_page_pdf"
    if image.width <= PARSE_INPUT_WH[0] and image.height <= PARSE_INPUT_WH[1]:
        return None, "already_within_model_input"
    candidate = render_retry_image(bytes(page_pdf))
    try:
        if parse35:
            same = effective_image_fingerprint(image, processor) == effective_image_fingerprint(candidate, processor)
        else:
            same = image.size == candidate.size and image.tobytes() == candidate.tobytes()
    except BaseException:
        candidate.close()
        raise
    if same:
        candidate.close()
        return None, "identical_effective_input" if parse35 else "identical_raster"
    return candidate, "changed_rendering"
