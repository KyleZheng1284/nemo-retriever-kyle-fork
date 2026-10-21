# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary tests use the real model-team parser and deterministic transport."""

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.models.nim import nemotron_parse as served
from nemo_retriever.models.nim import parse_rendering as rendering
from nemo_retriever.operators.extract.parse import nemotron_parse as stage
from nemo_retriever.operators.extract.pdf.extract import (
    build_pdf_extraction_kwargs,
    pdf_extraction,
)
from nemo_retriever.operators.extract.pdf.split import _split_pdf_to_single_page_bytes
from PIL import Image

RAW = "<x_0.1><y_0.1>Hello<x_0.9><y_0.2><class_Text>"
TABLE = "<x_0.1><y_0.1>| A | B |\n|---|---|\n| 1 | 2 |<x_0.9><y_0.9><class_Table>"
MODEL = "nvidia/Nemotron-3.5-Parse"
URL = "http://localhost:9999/v1/chat/completions"


def image_b64(color="white", size=(100, 120)):
    with Image.new("RGB", size, color) as image:
        return rendering.image_record(image)["image_b64"]


def response(raw=RAW, finish="stop"):
    return {"text": raw, "finish_reason": finish, "usage": {"completion_tokens": 10}}


class Client:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def invoke_chat_completions_images(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return [value]


def parse(client, **kwargs):
    pytest.importorskip("nemotron_parse.inference")
    return served.invoke_document_parse_images(
        invoke_url=URL,
        model=MODEL,
        image_b64_list=[image_b64()],
        page_pdfs=[b"pdf"],
        page_identities=[{"page_number": 7}],
        nim_client=client,
        **kwargs
    )[0]


@pytest.mark.parametrize(
    "mode,values,stages",
    [
        ("single", [response(TABLE)], ["fullpage"]),
        ("double", [response(TABLE), response(TABLE)], ["fullpage", "crop"]),
        ("triple", [response(TABLE), response(TABLE), response(TABLE)], ["fullpage", "detection", "crop"]),
    ],
)
def test_real_model_package_stages_and_settings(mode, values, stages):
    client = Client(values)
    result = parse(client, mode=mode)
    assert result["error"] is None
    assert [r["stage"] for r in result["attempts"]] == stages
    assert result["inference"]["mode"] == mode
    assert result["inference"]["merge_policy"] == "alternatives"
    for call, record in zip(client.calls, result["attempts"], strict=True):
        assert call["task_prompt"] == record["prompt"]
        assert record["page_identity"]["page_number"] == 7
        assert record["ended_unix"] >= record["started_unix"]
        assert record["input_image"]["sha256"]
        if record["stage"] == "detection":
            assert "<output_no_text>" in call["task_prompt"]
            assert call["extra_body"]["max_tokens"] == 4096
        elif record["stage"] == "crop":
            assert "<predict_text_in_pic>" in call["task_prompt"]
            assert call["temperature"] == 0.1
            assert call["repetition_penalty"] == 1.0
        else:
            assert call["temperature"] == 0.2
            assert call["repetition_penalty"] == 1.05


@pytest.mark.parametrize("first", [response("", "stop"), response("cut", "length")])
def test_retry_once_same_settings_and_selected_image(monkeypatch, first):
    monkeypatch.setattr(
        served, "prepare_retry", lambda *a, **k: (Image.new("RGB", (31, 37), "black"), "changed_rendering")
    )
    client = Client([first, response()])
    result = parse(client)
    assert result["error"] is None and result["selected_attempt"] == 1
    assert result["rendering_retry"]["selected"]
    assert len(client.calls) == 2
    a, b = result["attempts"]
    assert (a["prompt"], a["settings"]) == (b["prompt"], b["settings"])
    assert a["raw_output"] == first["text"]
    assert a["input_image"]["sha256"] != b["input_image"]["sha256"]
    assert result["selected_page_image"]["orig_shape_hw"] == (37, 31)
    assert a["attempt_id"] != b["attempt_id"]


@pytest.mark.parametrize(
    "last", [response("still cut", "length"), response("", "stop"), response(RAW, "eos"), RuntimeError("engine failed")]
)
def test_failed_retry_retains_original_and_both_attempts(monkeypatch, last):
    monkeypatch.setattr(served, "prepare_retry", lambda *a, **k: (Image.new("RGB", (30, 30)), "changed_rendering"))
    client = Client([response("cut", "length"), last])
    result = parse(client)
    assert result["error"] and result["selected_attempt"] == 0
    assert result["raw_output"] == "cut" and result["finish_reason"] == "length"
    assert len(result["attempts"]) == 2 and not result["rendering_retry"]["selected"]
    assert "selected_page_image" not in result


@pytest.mark.parametrize(
    "mode,values,stage_name",
    [
        ("single", [RuntimeError("transport")], "request"),
        ("single", [response("cancelled", "abort")], "fullpage"),
        ("double", [response(TABLE), response("", "stop")], "crop"),
        ("triple", [response(TABLE), response("cut", "length")], "detection"),
    ],
)
def test_noneligible_errors_never_trigger_retry(monkeypatch, mode, values, stage_name):
    monkeypatch.setattr(served, "prepare_retry", lambda *a, **k: pytest.fail("must not render"))
    client = Client(values)
    result = parse(client, mode=mode)
    assert result["error"]["stage"].endswith(stage_name)
    assert not result["rendering_retry"]["attempted"]
    assert len(result["attempts"]) == len(values)
    assert all(r["ended_unix"] is not None for r in result["attempts"])


def test_effectively_identical_retry_skips_request(monkeypatch):
    monkeypatch.setattr(served, "prepare_retry", lambda *a, **k: (None, "identical_effective_input"))
    client = Client([response("cut", "length")])
    result = parse(client)
    assert len(client.calls) == 1 and result["error"]
    assert result["rendering_retry"]["reason"] == "identical_effective_input"


def test_concurrent_pages_preserve_order_and_failed_siblings():
    pytest.importorskip("nemotron_parse.inference")
    barrier = threading.Barrier(2)

    class ConcurrentClient:
        def invoke_chat_completions_images(self, **kwargs):
            barrier.wait(timeout=5)
            with Image.open(io.BytesIO(base64.b64decode(kwargs["image_b64_list"][0]))) as im:
                white = im.getpixel((0, 0))[0] > 0
            return [response("cut" if white else RAW, "length" if white else "stop")]

    out = served.invoke_document_parse_images(
        invoke_url=URL,
        model=MODEL,
        nim_client=ConcurrentClient(),
        image_b64_list=[image_b64("white"), image_b64("black")],
        max_pool_workers=2,
        rendering_retry=False,
    )
    assert out[0]["error"] and out[1]["error"] is None
    assert out[1]["raw_output"] == RAW


def test_stage_drops_pdf_bytes_and_uses_retry_raster(monkeypatch):
    pytest.importorskip("nemotron_parse.inference")
    monkeypatch.setattr(served, "prepare_retry", lambda *a, **k: (Image.new("RGB", (31, 37)), "changed_rendering"))
    frame = pd.DataFrame(
        [
            {
                "path": "a.pdf",
                "page_number": 7,
                "page_image": {"image_b64": image_b64()},
                rendering.PAGE_PDF_COLUMN: b"pdf",
            }
        ]
    )
    out = stage.nemotron_parse_pages(
        frame,
        invoke_url=URL,
        nemotron_parse_model=MODEL,
        extract_text=True,
        nim_client=Client([response("cut", "length"), response()]),
    )
    assert rendering.PAGE_PDF_COLUMN not in out
    assert out.iloc[0]["page_number"] == 7 and out.iloc[0]["text"] == "Hello"
    assert out.iloc[0]["page_image"]["orig_shape_hw"] == (37, 31)
    assert out.iloc[0]["nemotron_parse_v1_2"]["attempts"][0]["page_identity"]["page_pdf_sha256"]
    assert rendering.PAGE_PDF_COLUMN in frame  # Caller input is not mutated.


def test_local_retry_retains_failed_pages_and_uses_same_prompt(monkeypatch):
    monkeypatch.setattr(stage, "prepare_retry", lambda *a, **k: (Image.new("RGB", (31, 37)), "changed_rendering"))
    calls = []

    class Local:
        generation_settings = {"temperature": 0.0}

        def invoke_batch_with_finish_reasons(self, images, **kwargs):
            calls.append((len(images), kwargs))
            return [("cut", "length"), (RAW, "stop")] if len(calls) == 1 else [(RAW, "stop")]

    frame = pd.DataFrame(
        [
            {"page_number": p, "page_image": {"image_b64": image_b64()}, rendering.PAGE_PDF_COLUMN: b"pdf"}
            for p in [1, 2]
        ]
    )
    out = stage.nemotron_parse_pages(frame, model=Local(), extract_text=True, nemotron_parse_rendering_retry=True)
    assert [x[0] for x in calls] == [2, 1]
    assert calls[0][1] == calls[1][1]
    assert out["text"].tolist() == ["Hello", "Hello"]
    assert out.iloc[0]["nemotron_parse_v1_2"]["selected_attempt"] == 1
    assert out.iloc[1]["nemotron_parse_v1_2"]["selected_attempt"] == 0
    assert rendering.PAGE_PDF_COLUMN not in out


@pytest.mark.parametrize("mode", ["double", "triple"])
def test_public_mode_validation(mode):
    with pytest.raises(ValueError, match="Multipass"):
        ExtractParams(method="nemotron_parse", nemotron_parse_mode=mode)
    p = ExtractParams(
        method="nemotron_parse", nemotron_parse_model=MODEL, nemotron_parse_invoke_url=URL, nemotron_parse_mode=mode
    )
    assert build_pdf_extraction_kwargs(p)["retain_page_pdf"] is True
    p.nemotron_parse_rendering_retry = False
    assert build_pdf_extraction_kwargs(p)["retain_page_pdf"] is False


def test_native_contract_does_not_auto_enable_retry():
    p = ExtractParams(
        method="nemotron_parse", nemotron_parse_model=MODEL, nemotron_parse_invoke_url="http://localhost:9999/v1/parse"
    )
    assert build_pdf_extraction_kwargs(p)["retain_page_pdf"] is False
    assert not rendering.retry_enabled(None, MODEL, p.nemotron_parse_invoke_url)


def make_pdf():
    pdfium = pytest.importorskip("pypdfium2")
    with rendering.PDFIUM_PROCESS_LOCK:
        doc = pdfium.PdfDocument.new()
        for width, height in [(300, 400), (400, 300)]:
            page = doc.new_page(width, height)
            page.close()
        buffer = io.BytesIO()
        doc.save(buffer)
        doc.close()
        return buffer.getvalue()


def test_real_pdf_identity_retention_concurrent_cleanup():
    raw = make_pdf()

    def run(_):
        pages = _split_pdf_to_single_page_bytes(raw)
        frame = pd.DataFrame([{"bytes": p, "path": "fixture.pdf", "page_number": i + 1} for i, p in enumerate(pages)])
        out = pdf_extraction(
            frame, extract_page_as_image=True, retain_page_pdf=True, image_format="png", dpi=72, render_mode="full_dpi"
        )
        assert out["page_number"].tolist() == [1, 2]
        assert [r["orig_shape_hw"] for r in out["page_image"]] == [(400, 300), (300, 400)]
        for pdf in out[rendering.PAGE_PDF_COLUMN]:
            image = rendering.render_retry_image(pdf)
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (255, 255, 255)
            image.close()
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(run, range(8)))
    with pytest.raises(ValueError, match="single-page"):
        rendering.render_retry_image(raw)


def test_effective_fingerprint_includes_geometry_and_skips_equal_input(monkeypatch):
    import numpy as np

    def processor(**kwargs):
        return {"pixel_values": np.zeros((1, 3, 2, 2)), "image_sizes": np.array([[2, 2]])}

    monkeypatch.setattr(rendering, "render_retry_image", lambda *a, **k: Image.new("RGB", (1664, 2048)))
    image = Image.new("RGB", (2000, 2500))
    candidate, reason = rendering.prepare_retry(image, b"pdf", parse35=True, processor=processor)
    assert candidate is None and reason == "identical_effective_input"

    def other(**kwargs):
        return {"pixel_values": np.zeros((1, 3, 2, 2)), "image_sizes": np.array([[1, 4]])}

    assert rendering.effective_image_fingerprint(image, processor) != rendering.effective_image_fingerprint(
        image, other
    )
    image.close()


def test_multipass_merged_result_reaches_interleaved_parquet(tmp_path):
    import pyarrow.parquet as pq
    from nemo_retriever.ingestor.parquet import ParquetDatasetWriter

    client = Client([response(TABLE), response(TABLE.replace("| 1 | 2 |", "| 7 | 8 |"))])
    frame = pd.DataFrame([{"path": "fixture.pdf", "page_number": 1, "page_image": {"image_b64": image_b64()}}])
    result = stage.nemotron_parse_pages(
        frame,
        invoke_url=URL,
        nemotron_parse_model=MODEL,
        nemotron_parse_mode="double",
        extract_text=True,
        extract_tables=True,
        nim_client=client,
    )
    meta = result.iloc[0]["nemotron_parse_v1_2"]
    assert "| 7 | 8 |" not in meta["raw_output"]
    assert "| 7 | 8 |" in meta["routing_output"]
    target = tmp_path / "delivered"
    ParquetDatasetWriter(str(target), output_schema="interleaved").write_batches([result])
    rows = pq.read_table(target).to_pylist()
    assert any("7" in (r["text_content"] or "") and "8" in (r["text_content"] or "") for r in rows)
    assert all(not r["materialize_error"] for r in rows)


def test_invalid_page_identity_does_not_shift_sibling_requests():
    client = Client([response(TABLE)])
    frame = pd.DataFrame(
        [
            {"path": "bad.pdf", "page_number": "not-a-page", "page_image": {"image_b64": image_b64()}},
            {"path": "good.pdf", "page_number": 7, "page_image": {"image_b64": image_b64()}},
        ]
    )
    result = stage.nemotron_parse_pages(
        frame, invoke_url=URL, nemotron_parse_model=MODEL, extract_text=True, nim_client=client
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"]
    second = result.iloc[1]["nemotron_parse_v1_2"]
    assert second["error"] is None
    assert second["attempts"][0]["page_identity"]["page_number"] == 7
    assert second["attempts"][0]["page_identity"]["path"] == "good.pdf"


def test_retry_processor_failure_closes_rendered_candidate(monkeypatch):
    candidate = Image.new("RGB", (1664, 2048))
    monkeypatch.setattr(rendering, "render_retry_image", lambda *a, **k: candidate)

    def fail(**kwargs):
        raise RuntimeError("processor unavailable")

    with Image.new("RGB", (2000, 2500)) as original:
        with pytest.raises(RuntimeError, match="processor unavailable"):
            rendering.prepare_retry(original, b"pdf", parse35=True, processor=fail)
    with pytest.raises(ValueError, match="closed"):
        candidate.getpixel((0, 0))


@pytest.mark.parametrize("mode", ["single", "double", "triple"])
def test_graph_entrypoints_share_parse_policy_and_render_before_inference(monkeypatch, mode):
    from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
    from nemo_retriever.graph.ingestor_runtime import build_graph
    from nemo_retriever.operators.graph_ops import multi_type_extract_operator as multi

    params = ExtractParams(
        method="nemotron_parse",
        nemotron_parse_model=MODEL,
        nemotron_parse_invoke_url=URL,
        nemotron_parse_mode=mode,
        nemotron_parse_triple_regions="detector",
        nemotron_parse_rendering_retry=True,
    )
    graph = build_graph(extraction_mode="pdf", extract_params=params)
    dedicated = get_node_kwargs(graph, "NemotronParseActor")
    render = get_node_kwargs(graph, "PDFExtractionActor")
    assert render["retain_page_pdf"] is True
    calls = []

    class Identity:
        def __init__(self, **kwargs):
            pass

        def run(self, data):
            return data

    class Render(Identity):
        def __init__(self, **kwargs):
            assert kwargs["extract_page_as_image"] and kwargs["retain_page_pdf"]

        def run(self, data):
            calls.append("render")
            return data.assign(page_image=[{"image_b64": image_b64()}])

    class Parse(Identity):
        def __init__(self, **kwargs):
            for key in ("nemotron_parse_mode", "nemotron_parse_triple_regions", "nemotron_parse_rendering_retry"):
                assert kwargs[key] == dedicated[key]

        def run(self, data):
            assert calls == ["render"] and data.iloc[0]["page_image"]["image_b64"]
            calls.append("parse")
            return data

    monkeypatch.setattr(multi, "DocToPdfConversionActor", Identity)
    monkeypatch.setattr(multi, "PDFSplitActor", Identity)
    monkeypatch.setattr(multi, "PDFExtractionActor", Render)
    operator = multi.MultiTypeExtractCPUActor(extraction_mode="pdf", extract_params=params)
    monkeypatch.setattr(operator, "_instantiate_resolved", lambda _cls, **kw: Parse(**kw))
    monkeypatch.setattr(operator, "_maybe_chunk", lambda data, _mode: data)
    operator._run_pdf_pipeline(pd.DataFrame([{"path": "fixture.pdf", "bytes": b"pdf"}]))
    assert calls == ["render", "parse"]


def test_retry_picture_projection_uses_selected_content_coordinates(monkeypatch):
    pytest.importorskip("nemotron_parse.inference")
    from nemo_retriever.ingestor.interleaved import project_page

    monkeypatch.setattr(
        served, "prepare_retry", lambda *a, **k: (Image.new("RGB", (220, 170), "green"), "changed_rendering")
    )
    picture = "<x_0.1><y_0.1>Figure<x_0.9><y_0.9><class_Picture>"
    frame = pd.DataFrame(
        [
            {
                "path": "landscape.pdf",
                "page_number": 7,
                "page_image": {"image_b64": image_b64("red"), "orig_shape_hw": [120, 100]},
                rendering.PAGE_PDF_COLUMN: b"pdf",
            }
        ]
    )
    result = (
        stage.nemotron_parse_pages(
            frame,
            invoke_url=URL,
            nemotron_parse_model=MODEL,
            nim_client=Client([response("cut", "length"), response(picture)]),
        )
        .iloc[0]
        .to_dict()
    )
    assert result["nemotron_parse_v1_2"]["selected_attempt"] == 1
    assert result["nemotron_parse_v1_2"]["bbox_coordinate_space"] == "normalized_page_content"
    (projected,) = project_page(result)
    assert projected["materialize_error"] is None
    assert projected["page_number"] == 6
    with Image.open(io.BytesIO(projected["binary_content"])) as crop:
        assert crop.size == (176, 136)
        assert crop.getextrema() == ((0, 0), (128, 128), (0, 0))
