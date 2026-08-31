# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import PropertyMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

import nemo_retriever.service.routers.ingest as ingest_module
import nemo_retriever.service.vectordb_app as vectordb_module
from nemo_retriever.service.app import create_app
from nemo_retriever.service.agentic_query import (
    agentic_ranked_to_hits,
    build_agentic_query_request,
    run_agentic_query,
)
from nemo_retriever.service.config import (
    AgenticConfig,
    AuthConfig,
    LoggingConfig,
    PipelinePoolConfig,
    ServiceConfig,
    VectorDbConfig,
)
from nemo_retriever.service.query_schema import (
    AgenticQueryResponse,
    MAX_AGENTIC_QUERY_CHARS,
    QueryRequest,
    QueryResult,
)
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def _progress_event(sequence: int = 1) -> dict[str, object]:
    return {
        "schema_version": "1",
        "run_id": "run-1",
        "sequence": sequence,
        "query_id": "query-1",
        "query_index": 0,
        "operation": "retrieval",
        "phase": "start",
        "message": "Starting initial retrieval.",
        "attributes": {"round": sequence, "kind": "initial", "requested": 10},
    }


def _request_for_app(app, *, payload: dict[str, object] | None = None, accept: str | None = None) -> Request:
    body = json.dumps(payload).encode() if payload is not None else b""
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if accept is not None:
        headers.append((b"accept", accept.encode()))
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "raw_path": b"/v1/query",
            "query_string": b"",
            "headers": headers,
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1234),
            "app": app,
        },
        receive,
    )
    request.state.authorized_scope = "default"
    return request


def _agentic_vectordb_app(tmp_path):
    return create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        reconciliation_interval_seconds=0,
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )


def _agentic_service_config(tmp_path, *, internal_api_token: str | None = None) -> ServiceConfig:
    return ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(
            enabled=True,
            vectordb_url="http://vectordb:7671",
            internal_api_token=internal_api_token,
        ),
        agentic=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )


def test_agentic_service_config_requires_remote_model_and_endpoint() -> None:
    with pytest.raises(ValidationError, match="agentic.invoke_url"):
        AgenticConfig(enabled=True, llm_model="model")
    with pytest.raises(ValidationError, match="agentic.llm_model"):
        AgenticConfig(
            enabled=True,
            invoke_url="https://llm.example/v1/chat/completions",
        )


def test_query_request_agentic_requires_hits_format() -> None:
    with pytest.raises(ValidationError, match="single query string"):
        QueryRequest(query=["a", "b"], agentic=True)
    with pytest.raises(ValidationError, match="non-empty"):
        QueryRequest(query="   ", agentic=True)
    with pytest.raises(ValidationError, match="format='hits'"):
        QueryRequest(query="q", agentic=True, format="evidence")


def test_build_agentic_query_request_maps_server_owned_configuration() -> None:
    request = build_agentic_query_request(
        query="revenue trend",
        top_k=3,
        config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
            backend_top_k=25,
            react_max_steps=7,
        ),
        lancedb_uri="/indexes/finance",
        table_name="finance",
        embed_endpoint="https://embed.example/v1/embeddings",
        embed_model="embed-model",
        embed_model_provider_prefix="openai",
        embed_api_key="embed-key",
    )

    assert request.query == "revenue trend"
    assert request.retrieval.top_k == 3
    assert request.storage.lancedb_uri == "/indexes/finance"
    assert request.storage.table_name == "finance"
    assert request.embed.embed_invoke_url == "https://embed.example/v1/embeddings"
    assert request.embed.embed_model_name == "embed-model"
    assert request.embed.embed_model_provider_prefix == "openai"
    assert request.embed.embed_api_key == "embed-key"
    assert request.agentic.enabled is True
    assert request.agentic.llm_model == "model"
    assert request.agentic.invoke_url == "https://llm.example/v1/chat/completions"
    assert request.agentic.backend_top_k == 25
    assert request.agentic.react_max_steps == 7


def test_run_agentic_query_includes_provider_usage() -> None:
    from nemo_retriever.query.workflow import AgenticQueryDocumentsResult

    usage = {
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
        "stages": {"main_agent": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}},
    }
    workflow_result = AgenticQueryDocumentsResult(
        hits=[{"doc_id": "report_7", "rank": 1, "result_source": "final_results"}],
        usage=usage,
    )

    def on_event(_event) -> None:
        pass

    with patch(
        "nemo_retriever.service.agentic_query.agentic_query_documents_with_metadata",
        return_value=workflow_result,
    ) as workflow:
        response = run_agentic_query(
            query="revenue trend",
            top_k=1,
            config=AgenticConfig(
                enabled=True,
                llm_model="model",
                invoke_url="https://llm.example/v1/chat/completions",
            ),
            lancedb_uri="/indexes/finance",
            table_name="finance",
            embed_endpoint="https://embed.example/v1/embeddings",
            embed_model="embed-model",
            embed_model_provider_prefix=None,
            embed_api_key="",
            on_event=on_event,
        )

    assert workflow.call_args.kwargs["on_event"] is on_event
    assert response.usage is not None
    assert response.usage.model_dump() == usage


def test_agentic_ranked_to_hits_keeps_rehydrated_classic_fields() -> None:
    hits = agentic_ranked_to_hits(
        [
            {
                "text": "revenue grew 4%",
                "metadata": {"type": "text"},
                "source": "/indexes/report.pdf",
                "source_id": "/indexes/report.pdf",
                "page_number": 7,
                "_score": 0.42,
                "rank": 1,
                "doc_id": "report_7",
                "result_source": "selection_agent",
            }
        ]
    )
    assert hits == [
        {
            "text": "revenue grew 4%",
            "metadata": {"type": "text", "rank": 1, "result_source": "selection_agent"},
            "source": "/indexes/report.pdf",
            "source_id": "/indexes/report.pdf",
            "page_number": 7,
            "_score": 0.42,
            "doc_id": "report_7",
            "rank": 1,
            "result_source": "selection_agent",
        }
    ]


def test_agentic_ranked_to_hits_falls_back_to_doc_id_without_rehydrated_metadata() -> None:
    hits = agentic_ranked_to_hits([{"rank": 1, "doc_id": "report.pdf", "result_source": "final_results"}])
    assert hits == [
        {
            "text": None,
            "metadata": {"rank": 1, "result_source": "final_results"},
            "source": "report.pdf",
            "source_id": None,
            "path": None,
            "page_number": None,
            "pdf_basename": None,
            "pdf_page": None,
            "doc_id": "report.pdf",
            "rank": 1,
            "result_source": "final_results",
        }
    ]


def test_agentic_ranked_to_hits_rejects_blank_doc_id() -> None:
    with pytest.raises(ValueError, match="missing a non-empty doc_id"):
        agentic_ranked_to_hits([{"rank": 1, "doc_id": "", "result_source": "selection_agent"}])


def test_agentic_query_flag_rejected_when_disabled_before_sse_streaming(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
    )

    with patch.object(vectordb_module, "run_agentic_query") as run_query, TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"query": "q", "agentic": True},
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "not enabled" in response.json()["detail"]
    run_query.assert_not_called()


def test_agentic_query_rejects_collection_target(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"query": "q", "agentic": True, "collection_name": "workspace"},
        )

    assert response.status_code == 501


def test_agentic_true_runs_react_workflow_on_v1_query(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="finance",
        embed_endpoint="https://embed.example/v1/embeddings",
        embed_model="embed-model",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )
    expected = AgenticQueryResponse(
        results=[
            QueryResult(
                hits=agentic_ranked_to_hits(
                    [
                        {
                            "text": "revenue grew 4%",
                            "metadata": {"type": "text"},
                            "source": "/indexes/report.pdf",
                            "page_number": 7,
                            "rank": 1,
                            "doc_id": "report_7",
                            "result_source": "selection_agent",
                        }
                    ]
                )
            )
        ],
        query_mode="agentic",
        usage={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "stages": {},
        },
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", return_value=expected) as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "top_k": 3, "agentic": True},
        )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "hits": [
                    {
                        "text": "revenue grew 4%",
                        "metadata": {"type": "text", "rank": 1, "result_source": "selection_agent"},
                        "source": "/indexes/report.pdf",
                        "page_number": 7,
                        "doc_id": "report_7",
                        "rank": 1,
                        "result_source": "selection_agent",
                    }
                ]
            }
        ],
        "query_mode": "agentic",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "stages": {},
        },
    }
    assert run_query.call_args.kwargs["query"] == "revenue trend"
    assert run_query.call_args.kwargs["top_k"] == 3
    assert run_query.call_args.kwargs["lancedb_uri"] == str(tmp_path)
    assert run_query.call_args.kwargs["table_name"] == "finance"
    assert run_query.call_args.kwargs["embed_api_key"] == ""
    assert run_query.call_args.kwargs["on_event"] is None


def test_agentic_query_sse_streams_progress_before_the_worker_completes(tmp_path) -> None:
    app = _agentic_vectordb_app(tmp_path)
    expected = AgenticQueryResponse(
        results=[QueryResult(hits=[])],
        query_mode="agentic",
        usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15, "stages": {}},
    )
    progress = _progress_event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    def run_with_progress(**kwargs):
        kwargs["on_event"](progress)
        assert release_worker.wait(timeout=5)
        worker_finished.set()
        return expected

    async def exercise_stream() -> tuple[object, str, str]:
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/query")
        async with app.router.lifespan_context(app):
            response = await endpoint(
                request=_request_for_app(app),
                req=QueryRequest(query="revenue trend", agentic=True),
                x_nrl_scope=None,
                accept="text/event-stream",
            )
            iterator = response.body_iterator.__aiter__()
            progress_record = await asyncio.wait_for(anext(iterator), timeout=5)
            assert worker_finished.is_set() is False
            release_worker.set()
            result_record = await asyncio.wait_for(anext(iterator), timeout=5)
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
        return response, progress_record, result_record

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", side_effect=run_with_progress),
    ):
        response, progress_record, result_record = asyncio.run(exercise_stream())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert isinstance(progress_record, str)
    assert progress_record.startswith("event: agentic_progress\ndata: ")
    assert json.loads(progress_record.split("\ndata: ", 1)[1]) == progress
    assert isinstance(result_record, str)
    assert result_record.startswith("event: result\ndata: ")
    assert json.loads(result_record.split("\ndata: ", 1)[1]) == expected.model_dump(mode="json")


def test_agentic_query_sse_disconnect_stops_enqueue_and_worker_releases_one_slot(tmp_path) -> None:
    app = _agentic_vectordb_app(tmp_path)
    expected = AgenticQueryResponse(results=[QueryResult(hits=[])])
    release_worker = threading.Event()
    worker_returned = threading.Event()

    def run_after_disconnect(**kwargs):
        kwargs["on_event"](_progress_event())
        assert release_worker.wait(timeout=5)
        kwargs["on_event"](_progress_event(sequence=2))
        worker_returned.set()
        return expected

    async def exercise_disconnect() -> None:
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/query")
        async with app.router.lifespan_context(app):
            loop = asyncio.get_running_loop()
            original_call_soon_threadsafe = loop.call_soon_threadsafe
            event_schedules = 0
            slot_released = threading.Event()
            slots = app.state.agentic_slots
            assert slots is not None
            original_release = slots.release

            def record_schedule(callback, *args):
                nonlocal event_schedules
                if getattr(callback, "__qualname__", "").endswith("schedule.<locals>.enqueue"):
                    event_schedules += 1
                return original_call_soon_threadsafe(callback, *args)

            def release_slot() -> None:
                original_release()
                slot_released.set()

            with (
                patch.object(loop, "call_soon_threadsafe", side_effect=record_schedule),
                patch.object(slots, "release", side_effect=release_slot) as release,
            ):
                response = await endpoint(
                    request=_request_for_app(app),
                    req=QueryRequest(query="revenue trend", agentic=True),
                    x_nrl_scope=None,
                    accept="text/event-stream",
                )
                iterator = response.body_iterator.__aiter__()
                first_record = await asyncio.wait_for(anext(iterator), timeout=5)
                assert "event: agentic_progress" in first_record

                await iterator.aclose()
                schedules_at_disconnect = event_schedules
                release_worker.set()
                assert await asyncio.to_thread(worker_returned.wait, 5)
                assert await asyncio.to_thread(slot_released.wait, 5)
                await asyncio.sleep(0)
                release.assert_called_once_with()
                assert event_schedules == schedules_at_disconnect

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", side_effect=run_after_disconnect),
    ):
        asyncio.run(exercise_disconnect())


def test_agentic_query_openapi_advertises_json_and_sse(tmp_path) -> None:
    vectordb_app = create_vectordb_app(lancedb_uri=str(tmp_path))
    service_app = create_app(_agentic_service_config(tmp_path))

    vectordb_content = vectordb_app.openapi()["paths"]["/v1/query"]["post"]["responses"]["200"]["content"]
    service_content = service_app.openapi()["paths"]["/v1/query"]["post"]["responses"]["200"]["content"]

    assert "application/json" in vectordb_content
    assert vectordb_content["text/event-stream"]["schema"] == {"type": "string"}
    assert "application/json" in service_content
    assert service_content["text/event-stream"]["schema"] == {"type": "string"}


def test_agentic_query_sse_sanitizes_worker_failure(tmp_path) -> None:
    app = _agentic_vectordb_app(tmp_path)

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", side_effect=RuntimeError("private endpoint")),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "agentic": True},
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    assert response.text == (
        "event: error\n" 'data: {"code":"agentic_query_failed","message":"Agentic retrieval failed."}\n\n'
    )
    assert "private endpoint" not in response.text


def test_agentic_query_rejects_top_k_above_backend_depth(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
            backend_top_k=5,
        ),
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "top_k": 6, "agentic": True},
        )

    assert response.status_code == 422
    assert "cannot exceed" in response.json()["detail"]


def test_agentic_query_rejects_query_above_length_limit(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query") as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            json={"query": "x" * (MAX_AGENTIC_QUERY_CHARS + 1), "agentic": True},
        )

    assert response.status_code == 422
    run_query.assert_not_called()


def test_agentic_query_slots_are_bounded_and_released_by_the_worker(tmp_path) -> None:
    """Capacity follows the worker thread, not the caller: a saturated pool sheds
    load with 503 instead of queueing behind non-cancellable ReAct work, and a
    completed query returns its slot."""
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )
    expected = AgenticQueryResponse(results=[QueryResult(hits=[])])

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", return_value=expected),
        TestClient(app) as client,
    ):
        slots = app.state.agentic_slots
        assert slots is not None

        for _ in range(vectordb_module.MAX_CONCURRENT_AGENTIC_QUERIES):
            assert slots.acquire(blocking=False) is True

        busy = client.post("/v1/query", json={"query": "revenue trend", "agentic": True})

        assert busy.status_code == 503
        assert busy.headers["Retry-After"] == "30"
        query_semaphore = app.state.vectordb_state.query_semaphore
        assert query_semaphore.locked() is False

        slots.release()
        accepted = client.post("/v1/query", json={"query": "revenue trend", "agentic": True})

        assert accepted.status_code == 200
        assert slots.acquire(blocking=False) is True


def test_gateway_proxies_agentic_flag_to_vectordb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def _stub_work(_item):
        return 0, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _config: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _config: _stub_work,
    )
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(
            enabled=True,
            vectordb_url="http://vectordb:7671",
        ),
        agentic=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
            request_timeout_s=321.0,
        ),
    )
    seen: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        content = json.dumps({"results": [{"hits": []}]}).encode()

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            seen["timeout"] = kwargs["timeout"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            seen["url"] = url
            seen["body"] = json.loads(kwargs["content"])
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "top_k": 3, "agentic": True},
        )

    assert response.status_code == 200
    assert response.json() == {"results": [{"hits": []}]}
    assert seen == {
        "timeout": 321.0,
        "url": "http://vectordb:7671/v1/query",
        "body": {"query": "revenue trend", "top_k": 3, "agentic": True},
    }


def test_service_rejects_agentic_flag_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def _stub_work(_item):
        return 0, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _config: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _config: _stub_work,
    )
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
    )

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "agentic": True},
        )

    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"]


def test_gateway_streams_first_agentic_sse_chunk_before_upstream_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = _agentic_service_config(tmp_path, internal_api_token="internal-token")
    chunks = (
        b'event: agentic_progress\ndata: {"sequence":1}\n\n',
        b'event: result\ndata: {"results":[{"hits":[]}]}\n\n',
    )
    seen: dict[str, object] = {}

    async def exercise_stream() -> tuple[object, bytes, bytes]:
        release_upstream = asyncio.Event()
        upstream_blocked = asyncio.Event()

        class _FakeResponse:
            status_code = 200
            headers = {
                "content-type": "text/event-stream; charset=utf-8",
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            }

            async def aiter_raw(self):
                yield chunks[0]
                upstream_blocked.set()
                await release_upstream.wait()
                yield chunks[1]
                seen["upstream_finished"] = True

            async def aclose(self) -> None:
                seen["response_closed"] = True

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs) -> None:
                seen["timeout"] = kwargs["timeout"]

            def build_request(self, method: str, url: str, **kwargs):
                seen["method"] = method
                seen["url"] = url
                seen["body"] = json.loads(kwargs["content"])
                seen["headers"] = kwargs["headers"]
                return object()

            async def send(self, _request, *, stream: bool):
                seen["stream"] = stream
                return _FakeResponse()

            async def aclose(self) -> None:
                seen["client_closed"] = True

        monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
        app = create_app(config)
        response = await ingest_module.query(
            _request_for_app(
                app,
                payload={"query": "revenue trend", "agentic": True},
                accept="text/event-stream",
            )
        )
        iterator = response.body_iterator.__aiter__()
        first_chunk = await anext(iterator)
        assert "upstream_finished" not in seen
        second_chunk_task = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(upstream_blocked.wait(), timeout=5)
        assert second_chunk_task.done() is False
        release_upstream.set()
        second_chunk = await asyncio.wait_for(second_chunk_task, timeout=5)
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        return response, first_chunk, second_chunk

    response, first_chunk, second_chunk = asyncio.run(exercise_stream())

    assert response.status_code == 200
    assert first_chunk == chunks[0]
    assert second_chunk == chunks[1]
    assert seen["method"] == "POST"
    assert seen["url"] == "http://vectordb:7671/v1/query"
    assert seen["body"] == {"query": "revenue trend", "agentic": True}
    assert seen["headers"] == {
        "Content-Type": "application/json",
        "X-NRL-Scope": "default",
        "X-NRL-Internal-Token": "internal-token",
        "Accept": "text/event-stream",
    }
    assert seen["stream"] is True
    assert seen["upstream_finished"] is True
    assert seen["response_closed"] is True
    assert seen["client_closed"] is True


def test_gateway_buffers_mixed_version_json_for_agentic_sse_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = _agentic_service_config(tmp_path)
    payload = {
        "results": [{"hits": []}],
        "query_mode": "agentic",
        "usage": None,
    }
    seen: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aread(self) -> bytes:
            seen["read"] = True
            return json.dumps(payload).encode()

        async def aclose(self) -> None:
            seen["response_closed"] = True

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            seen["timeout"] = kwargs["timeout"]

        def build_request(self, _method: str, _url: str, **kwargs):
            seen["accept"] = kwargs["headers"].get("Accept")
            return object()

        async def send(self, _request, *, stream: bool):
            seen["stream"] = stream
            return _FakeResponse()

        async def aclose(self) -> None:
            seen["client_closed"] = True

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    app = create_app(config)
    response = asyncio.run(
        ingest_module.query(
            _request_for_app(
                app,
                payload={"query": "revenue trend", "agentic": True},
                accept="text/event-stream",
            )
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert json.loads(response.body) == payload
    assert seen == {
        "timeout": 1800.0,
        "accept": "text/event-stream",
        "stream": True,
        "read": True,
        "response_closed": True,
        "client_closed": True,
    }
