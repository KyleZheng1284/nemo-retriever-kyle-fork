# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import nemo_retriever.service.vectordb_app as vectordb_module
from nemo_retriever.common.vdb.adt_vdb import VDBInvalidRequest, VDBResourceNotFound
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
    MAX_AGENTIC_QUERY_CHARS,
    QueryRequest,
    QueryResponse,
    QueryResult,
)
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def _create_enabled_agentic_app(tmp_path, **kwargs):
    return create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
        **kwargs,
    )


def test_agentic_service_config_requires_remote_model_and_endpoint() -> None:
    assert AgenticConfig().max_tokens == 1024
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        AgenticConfig(max_tokens=0)
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
            max_tokens=2048,
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
    assert request.agentic.max_tokens == 2048


def test_vectordb_main_maps_agentic_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vectordb_app",
            "--agentic",
            "--agentic-llm-model",
            "model",
            "--agentic-invoke-url",
            "https://llm.example/v1/chat/completions",
            "--agentic-max-tokens",
            "2048",
        ],
    )
    create_app = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(vectordb_module, "create_vectordb_app", create_app)
    monkeypatch.setattr(vectordb_module.uvicorn, "run", MagicMock())

    vectordb_module.main()

    assert create_app.call_args.kwargs["agentic_config"].max_tokens == 2048


def test_run_agentic_query_threads_canonical_hit_callback() -> None:
    retrieve_hits = MagicMock()
    config = AgenticConfig(
        enabled=True,
        llm_model="model",
        invoke_url="https://llm.example/v1/chat/completions",
    )

    with patch(
        "nemo_retriever.service.agentic_query.agentic_query_documents",
        return_value=[],
    ) as query_documents:
        response = run_agentic_query(
            query="revenue trend",
            top_k=3,
            config=config,
            lancedb_uri="/indexes/finance",
            table_name="finance",
            embed_endpoint="https://embed.example/v1/embeddings",
            embed_model="embed-model",
            embed_model_provider_prefix="openai",
            embed_api_key="embed-key",
            retrieve_hits_fn=retrieve_hits,
            doc_id_field="chunk_id",
        )

    assert response.query_mode == "agentic"
    workflow_request = query_documents.call_args.args[0]
    assert workflow_request.query == "revenue trend"
    query_documents.assert_called_once_with(
        workflow_request,
        retrieve_hits_fn=retrieve_hits,
        doc_id_field="chunk_id",
    )


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


def test_agentic_query_flag_rejected_when_disabled(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
    )

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "q", "agentic": True})

    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"]


def test_agentic_query_binds_each_scope_to_its_logical_collection(tmp_path) -> None:
    backend = MagicMock()

    def retrieve_collection(_vectors, *, scope, collection_name, query_texts, top_k, **_kwargs):
        return (
            [
                [
                    {
                        "chunk_id": f"{scope}-chunk",
                        "document_id": f"{scope}-document",
                        "text": f"hit for {scope}",
                        "distance": 0.2,
                        "filename": f"{scope}.pdf",
                        "metadata": {},
                        "physical_table": f"private-{scope}",
                    }
                ]
            ],
            ["dense"],
        )

    backend.retrieve_collection.side_effect = retrieve_collection
    app = _create_enabled_agentic_app(
        tmp_path,
        vdb=backend,
        reconciliation_interval_seconds=0,
    )

    def fake_agentic_query(**kwargs):
        hits = kwargs["retrieve_hits_fn"]("rewritten query", 7)
        selected = dict(hits[0])
        selected.update(
            {
                "doc_id": selected["chunk_id"],
                "rank": 1,
                "result_source": "selection_agent",
            }
        )
        return QueryResponse(
            results=[QueryResult(hits=[selected])],
            query_mode="agentic",
        )

    with (
        patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]) as embed_queries,
        patch.object(vectordb_module, "run_agentic_query", side_effect=fake_agentic_query) as run_query,
        TestClient(app) as client,
    ):
        responses = [
            client.post(
                "/v1/query",
                headers={"X-NRL-Scope": scope},
                json={"query": "q", "agentic": True, "collection_name": "workspace"},
            )
            for scope in ("tenant-a", "tenant-b")
        ]

    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["results"][0]["hits"][0]["text"] for response in responses] == [
        "hit for tenant-a",
        "hit for tenant-b",
    ]
    for response, scope in zip(responses, ("tenant-a", "tenant-b")):
        hit = response.json()["results"][0]["hits"][0]
        assert hit["doc_id"] == f"{scope}-chunk"
        assert hit["document_id"] == f"{scope}-document"
        assert "physical_table" not in hit
    assert [call.kwargs for call in backend.get_collection.call_args_list] == [
        {"scope": "tenant-a", "collection_name": "workspace"},
        {"scope": "tenant-b", "collection_name": "workspace"},
    ]
    assert [call.kwargs for call in backend.retrieve_collection.call_args_list] == [
        {
            "scope": "tenant-a",
            "collection_name": "workspace",
            "query_texts": ["rewritten query"],
            "top_k": 7,
        },
        {
            "scope": "tenant-b",
            "collection_name": "workspace",
            "query_texts": ["rewritten query"],
            "top_k": 7,
        },
    ]
    assert [call.args for call in backend.retrieve_collection.call_args_list] == [
        ([[1.0, 0.0]],),
        ([[1.0, 0.0]],),
    ]
    assert embed_queries.call_args_list == [call(["rewritten query"]), call(["rewritten query"])]
    assert all(call.kwargs["doc_id_field"] == "chunk_id" for call in run_query.call_args_list)
    backend.health.assert_not_called()


def test_agentic_collection_query_preflights_before_starting_agent(tmp_path) -> None:
    backend = MagicMock()
    backend.get_collection.side_effect = VDBResourceNotFound("Collection not found")
    app = _create_enabled_agentic_app(
        tmp_path,
        vdb=backend,
        reconciliation_interval_seconds=0,
    )

    with (
        patch.object(VectorDBState, "embed_queries") as embed_queries,
        patch.object(vectordb_module, "run_agentic_query") as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            headers={"X-NRL-Scope": "tenant-a"},
            json={"query": "q", "agentic": True, "collection_name": "missing"},
        )

    assert response.status_code == 404
    backend.get_collection.assert_called_once_with(scope="tenant-a", collection_name="missing")
    backend.health.assert_not_called()
    embed_queries.assert_not_called()
    run_query.assert_not_called()


@pytest.mark.parametrize(
    ("retrieval_error", "expected_status"),
    [
        pytest.param(VDBInvalidRequest("Collection is deleting"), 422, id="deleting"),
        pytest.param(VDBInvalidRequest("Collection is expired"), 422, id="expired"),
        pytest.param(VDBResourceNotFound("Collection not found"), 404, id="deleted-after-preflight"),
    ],
)
def test_agentic_collection_query_restores_lifecycle_error_after_agent_failure(
    tmp_path,
    retrieval_error: Exception,
    expected_status: int,
) -> None:
    backend = MagicMock()
    backend.retrieve_collection.side_effect = retrieval_error
    app = _create_enabled_agentic_app(
        tmp_path,
        vdb=backend,
        reconciliation_interval_seconds=0,
    )

    def fail_after_retrieval(**kwargs):
        try:
            kwargs["retrieve_hits_fn"]("rewritten query", 7)
        except (VDBInvalidRequest, VDBResourceNotFound):
            raise RuntimeError("Agentic retrieval tool failed") from None
        raise AssertionError("Expected collection retrieval to fail")

    with (
        patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]) as embed_queries,
        patch.object(vectordb_module, "run_agentic_query", side_effect=fail_after_retrieval) as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            headers={"X-NRL-Scope": "tenant-a"},
            json={"query": "q", "agentic": True, "collection_name": "workspace"},
        )

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(retrieval_error)}
    backend.get_collection.assert_called_once_with(scope="tenant-a", collection_name="workspace")
    backend.retrieve_collection.assert_called_once_with(
        [[1.0, 0.0]],
        scope="tenant-a",
        collection_name="workspace",
        query_texts=["rewritten query"],
        top_k=7,
    )
    backend.health.assert_not_called()
    embed_queries.assert_called_once_with(["rewritten query"])
    run_query.assert_called_once()


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
    expected = QueryResponse(
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
    }
    assert run_query.call_args.kwargs["query"] == "revenue trend"
    assert run_query.call_args.kwargs["top_k"] == 3
    assert run_query.call_args.kwargs["lancedb_uri"] == str(tmp_path)
    assert run_query.call_args.kwargs["table_name"] == "finance"
    assert run_query.call_args.kwargs["embed_api_key"] == ""


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
    app = _create_enabled_agentic_app(tmp_path)

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
    app = _create_enabled_agentic_app(tmp_path)
    expected = QueryResponse(results=[QueryResult(hits=[])], query_mode="agentic")

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
        auth=AuthConfig(
            enabled=True,
            api_token="public-secret",
            default_scope="tenant-a",
            allow_unscoped_dev=False,
        ),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(
            enabled=True,
            vectordb_url="http://vectordb:7671",
            internal_api_token="internal-secret",
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
            seen["headers"] = kwargs["headers"]
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer public-secret"},
            json={"query": "revenue trend", "top_k": 3, "agentic": True},
        )

    assert response.status_code == 200
    assert response.json() == {"results": [{"hits": []}]}
    assert seen["timeout"] == 321.0
    assert seen["url"] == "http://vectordb:7671/v1/query"
    assert seen["body"] == {"query": "revenue trend", "top_k": 3, "agentic": True}
    forwarded_headers = seen["headers"]
    assert isinstance(forwarded_headers, dict)
    assert forwarded_headers["Content-Type"] == "application/json"
    assert forwarded_headers["X-NRL-Scope"] == "tenant-a"
    assert forwarded_headers["X-NRL-Internal-Token"] == "internal-secret"
    assert "Authorization" not in forwarded_headers


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
