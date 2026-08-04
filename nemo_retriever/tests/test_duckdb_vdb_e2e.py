# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end VDB contract coverage with a non-LanceDB storage engine."""

from __future__ import annotations

import base64
import json
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
    IngestOperation,
)
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    VDB,
    VDBInvalidRequest,
    VDBResourceConflict,
    VDBResourceNotFound,
)
from nemo_retriever.common.vdb.records import RetrievalContractError
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def _encode_cursor(
    resource: str,
    scope: str,
    collection_name: str | None,
    last: list[str],
) -> str:
    payload = _json_dump(
        {
            "resource": resource,
            "scope": scope,
            "collection_name": collection_name,
            "last": last,
        }
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(
    token: str | None,
    *,
    resource: str,
    scope: str,
    collection_name: str | None,
) -> list[str] | None:
    if token is None:
        return None
    try:
        padding = "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(token + padding))
        if (
            not isinstance(payload, dict)
            or payload.get("resource") != resource
            or payload.get("scope") != scope
            or payload.get("collection_name") != collection_name
            or not isinstance(payload.get("last"), list)
            or any(not isinstance(value, str) for value in payload["last"])
        ):
            raise ValueError("cursor context mismatch")
        return payload["last"]
    except Exception as exc:
        raise VDBInvalidRequest(f"Invalid {resource} continuation token") from exc


class _DuckDBVDB(VDB):
    """Test-only VDB proving the public contracts do not depend on LanceDB."""

    def __init__(self, *, vector_dim: int = 2) -> None:
        if vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        self.vector_dim = int(vector_dim)
        self.hybrid = False
        self._lock = threading.RLock()
        self._connection = duckdb.connect(":memory:")
        self._legacy_sequence = 0
        super().__init__(vector_dim=vector_dim)
        self._create_schema()

    @property
    def _vector_type(self) -> str:
        return f"FLOAT[{self.vector_dim}]"

    def close(self) -> None:
        """Release the in-memory database after a test."""
        with self._lock:
            self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _create_schema(self) -> None:
        """Create backend-owned legacy, catalog, document, and chunk tables."""
        vector_type = self._vector_type
        with self._lock:
            self._connection.execute(
                f"""
                CREATE TABLE legacy_chunks (
                    chunk_id VARCHAR PRIMARY KEY,
                    embedding {vector_type} NOT NULL,
                    text VARCHAR NOT NULL,
                    metadata_json VARCHAR NOT NULL,
                    source_id VARCHAR NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE collections (
                    scope VARCHAR NOT NULL,
                    name VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    description VARCHAR,
                    metadata_json VARCHAR NOT NULL,
                    created_at VARCHAR NOT NULL,
                    updated_at VARCHAR NOT NULL,
                    expires_at VARCHAR,
                    PRIMARY KEY (scope, name)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE documents (
                    scope VARCHAR NOT NULL,
                    collection_name VARCHAR NOT NULL,
                    document_id VARCHAR NOT NULL,
                    filename VARCHAR NOT NULL,
                    content_sha256 VARCHAR NOT NULL,
                    document_version VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    job_id VARCHAR,
                    created_at VARCHAR NOT NULL,
                    updated_at VARCHAR NOT NULL,
                    error VARCHAR,
                    PRIMARY KEY (scope, collection_name, document_id)
                )
                """
            )
            self._connection.execute(
                f"""
                CREATE TABLE collection_chunks (
                    scope VARCHAR NOT NULL,
                    collection_name VARCHAR NOT NULL,
                    document_id VARCHAR NOT NULL,
                    chunk_id VARCHAR NOT NULL,
                    document_version VARCHAR NOT NULL,
                    content_sha256 VARCHAR NOT NULL,
                    filename VARCHAR NOT NULL,
                    embedding {vector_type} NOT NULL,
                    text VARCHAR NOT NULL,
                    page_number INTEGER,
                    content_type VARCHAR,
                    source_id VARCHAR,
                    stored_image_uri VARCHAR,
                    bbox_json VARCHAR,
                    metadata_json VARCHAR NOT NULL,
                    PRIMARY KEY (scope, collection_name, chunk_id)
                )
                """
            )

    def _fetch_dicts(self, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self._connection.execute(sql, parameters or [])
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _fetch_one(self, sql: str, parameters: list[Any] | None = None) -> dict[str, Any] | None:
        rows = self._fetch_dicts(sql, parameters)
        return rows[0] if rows else None

    def _validated_vector(self, value: Any) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != self.vector_dim:
            raise VDBInvalidRequest(f"Expected a {self.vector_dim}-dimensional embedding")
        vector = [float(item) for item in value]
        if any(not math.isfinite(item) for item in vector):
            raise VDBInvalidRequest("Embedding values must be finite")
        return vector

    def _record_values(self, record: dict[str, Any]) -> dict[str, Any] | None:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("embedding") is None:
            return None
        text = metadata.get("content")
        if not isinstance(text, str):
            return None
        content_metadata = metadata.get("content_metadata")
        content_metadata = dict(content_metadata) if isinstance(content_metadata, dict) else {}
        source_metadata = metadata.get("source_metadata")
        source_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        content_type = str(content_metadata.get("type") or record.get("document_type") or "text")
        if not text.strip() and content_type != "image":
            return None
        source_id = str(source_metadata.get("source_id") or source_metadata.get("source_name") or "")
        page_number = content_metadata.get("page_number")
        if page_number is not None:
            try:
                page_number = int(page_number)
            except (TypeError, ValueError) as exc:
                raise VDBInvalidRequest("page_number must be an integer") from exc
        bbox = content_metadata.get("bbox_xyxy_norm")
        return {
            "embedding": self._validated_vector(metadata["embedding"]),
            "text": text,
            "metadata_json": _json_dump(content_metadata),
            "source_id": source_id,
            "page_number": page_number,
            "content_type": content_type,
            "stored_image_uri": str(content_metadata.get("stored_image_uri") or "") or None,
            "bbox_json": _json_dump(bbox) if bbox is not None else None,
        }

    def _record_batches(self, records: list) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for batch in records:
            if not isinstance(batch, list):
                continue
            for record in batch:
                if isinstance(record, dict) and (row := self._record_values(record)) is not None:
                    values.append(row)
        return values

    def create_index(self, **kwargs: Any) -> None:
        """The exact-search test backend needs no secondary index."""
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        """Append canonical records to the backend's legacy fixed table."""
        rows = self._record_batches(records)
        with self._lock, self._transaction():
            for row in rows:
                chunk_id = f"legacy-{self._legacy_sequence:08d}"
                self._legacy_sequence += 1
                self._connection.execute(
                    """
                    INSERT INTO legacy_chunks
                        (chunk_id, embedding, text, metadata_json, source_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        chunk_id,
                        row["embedding"],
                        row["text"],
                        row["metadata_json"],
                        row["source_id"],
                    ],
                )

    def retrieval(self, queries: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Rank legacy rows with DuckDB's exact cosine-distance function."""
        top_k = int(kwargs.get("top_k", 10))
        results: list[list[dict[str, Any]]] = []
        with self._lock:
            for query in queries:
                vector = self._validated_vector(query)
                rows = self._fetch_dicts(
                    f"""
                    SELECT
                        chunk_id,
                        text,
                        metadata_json,
                        source_id,
                        array_cosine_distance(embedding, ?::{self._vector_type}) AS distance
                    FROM legacy_chunks
                    ORDER BY distance ASC, chunk_id ASC
                    LIMIT ?
                    """,
                    [vector, top_k],
                )
                results.append(
                    [
                        {
                            "chunk_id": row["chunk_id"],
                            "text": row["text"],
                            "metadata": _json_dict(row["metadata_json"]),
                            "source": row["source_id"],
                            "_distance": float(row["distance"]),
                        }
                        for row in rows
                    ]
                )
        return results

    def run(self, records: list) -> None:
        """Ensure the exact-search table exists, then write the batch."""
        self.create_index()
        self.write_to_index(records)

    def health(self) -> dict[str, Any]:
        """Expose the backend-neutral fields consumed by the VectorDB service."""
        with self._lock:
            total_rows = int(self._connection.execute("SELECT count(*) FROM legacy_chunks").fetchone()[0])
            active_collections = int(
                self._connection.execute("SELECT count(*) FROM collections WHERE status = 'active'").fetchone()[0]
            )
        return {
            "total_rows": total_rows,
            "table_exists": total_rows > 0,
            "effective_retrieval_mode": "dense" if total_rows else None,
            "retrieval_strategies": ["dense"] if total_rows else [],
            "collections": {"active": active_collections, "deleting": 0, "expired": 0},
            "cleanup": {"pending": 0, "oldest_age_seconds": 0},
            "reconciliation": {"successes": 0, "failures": 0},
            "open_table_cache_count": 0,
        }

    @staticmethod
    def _collection_info(row: dict[str, Any]) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"],
            scope=row["scope"],
            status=row["status"],
            description=row["description"],
            metadata=_json_dict(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    @staticmethod
    def _document_info(row: dict[str, Any]) -> DocumentInfo:
        return DocumentInfo(**{key: row.get(key) for key in DocumentInfo.model_fields})

    def _collection_row(self, scope: str, collection_name: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM collections WHERE scope = ? AND name = ?",
            [scope, collection_name],
        )

    def _require_collection(self, scope: str, collection_name: str) -> dict[str, Any]:
        row = self._collection_row(scope, collection_name)
        if row is None or row["status"] != "active":
            raise VDBResourceNotFound("Collection not found")
        return row

    def create_collection(self, *, scope: str, request: CollectionCreateRequest) -> CollectionInfo:
        """Create an independently scoped DuckDB catalog row."""
        with self._lock:
            if self._collection_row(scope, request.name) is not None:
                raise VDBResourceConflict(f"Collection {request.name!r} already exists")
            now = _now()
            self._connection.execute(
                """
                INSERT INTO collections
                    (scope, name, status, description, metadata_json, created_at, updated_at, expires_at)
                VALUES (?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                [
                    scope,
                    request.name,
                    request.description,
                    _json_dump(request.metadata),
                    now,
                    now,
                    request.expires_at,
                ],
            )
            return self.get_collection(scope=scope, collection_name=request.name)

    def get_collection(self, *, scope: str, collection_name: str) -> CollectionInfo:
        """Return one logical collection without exposing its storage rows."""
        with self._lock:
            return self._collection_info(self._require_collection(scope, collection_name))

    def list_collections(
        self,
        *,
        scope: str,
        limit: int,
        continuation_token: str | None,
    ) -> CollectionPage:
        """List scoped collections with a context-bound keyset cursor."""
        last = _decode_cursor(
            continuation_token,
            resource="collections",
            scope=scope,
            collection_name=None,
        )
        if last is not None and len(last) != 1:
            raise VDBInvalidRequest("Invalid collections continuation token")
        sql = "SELECT * FROM collections WHERE scope = ? AND status = 'active'"
        parameters: list[Any] = [scope]
        if last:
            sql += " AND name > ?"
            parameters.append(last[0])
        sql += " ORDER BY name ASC LIMIT ?"
        parameters.append(limit + 1)
        with self._lock:
            rows = self._fetch_dicts(sql, parameters)
        page = rows[:limit]
        next_token = None
        if len(rows) > limit and page:
            next_token = _encode_cursor("collections", scope, None, [page[-1]["name"]])
        return CollectionPage(items=[self._collection_info(row) for row in page], next_token=next_token)

    def update_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        request: CollectionUpdateRequest,
    ) -> CollectionInfo:
        """Update only the mutable collection fields defined by the contract."""
        with self._lock:
            row = self._require_collection(scope, collection_name)
            supplied = request.model_dump(exclude_unset=True)
            description = supplied.get("description", row["description"])
            metadata_json = row["metadata_json"]
            if "metadata" in supplied:
                metadata_json = _json_dump(supplied["metadata"] or {})
            expires_at = supplied.get("expires_at", row["expires_at"])
            self._connection.execute(
                """
                UPDATE collections
                SET description = ?, metadata_json = ?, expires_at = ?, updated_at = ?
                WHERE scope = ? AND name = ?
                """,
                [
                    description,
                    metadata_json,
                    expires_at,
                    _now(),
                    scope,
                    collection_name,
                ],
            )
            return self.get_collection(scope=scope, collection_name=collection_name)

    def delete_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        if_exists: bool,
    ) -> CollectionDeleteResult:
        """Synchronously delete a collection and its backend-owned rows."""
        with self._lock:
            if self._collection_row(scope, collection_name) is None:
                if if_exists:
                    return CollectionDeleteResult(
                        name=collection_name,
                        scope=scope,
                        existed=False,
                        deleted=False,
                        status="deleted",
                    )
                raise VDBResourceNotFound("Collection not found")
            with self._transaction():
                self._connection.execute(
                    "DELETE FROM collection_chunks WHERE scope = ? AND collection_name = ?",
                    [scope, collection_name],
                )
                self._connection.execute(
                    "DELETE FROM documents WHERE scope = ? AND collection_name = ?",
                    [scope, collection_name],
                )
                self._connection.execute(
                    "DELETE FROM collections WHERE scope = ? AND name = ?",
                    [scope, collection_name],
                )
        return CollectionDeleteResult(
            name=collection_name,
            scope=scope,
            existed=True,
            deleted=True,
            status="deleted",
        )

    def _document_row(self, scope: str, collection_name: str, document_id: str) -> dict[str, Any] | None:
        return self._fetch_one(
            """
            SELECT * FROM documents
            WHERE scope = ? AND collection_name = ? AND document_id = ?
            """,
            [scope, collection_name, document_id],
        )

    def get_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
    ) -> DocumentInfo:
        """Return one committed document from a scoped collection."""
        with self._lock:
            self._require_collection(scope, collection_name)
            row = self._document_row(scope, collection_name, document_id)
            if row is None:
                raise VDBResourceNotFound("Document not found")
            return self._document_info(row)

    def list_documents(
        self,
        *,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> DocumentPage:
        """List scoped documents with a collection-bound keyset cursor."""
        last = _decode_cursor(
            continuation_token,
            resource="documents",
            scope=scope,
            collection_name=collection_name,
        )
        if last is not None and len(last) != 2:
            raise VDBInvalidRequest("Invalid documents continuation token")
        sql = "SELECT * FROM documents WHERE scope = ? AND collection_name = ?"
        parameters: list[Any] = [scope, collection_name]
        if last:
            sql += " AND (created_at > ? OR (created_at = ? AND document_id > ?))"
            parameters.extend([last[0], last[0], last[1]])
        sql += " ORDER BY created_at ASC, document_id ASC LIMIT ?"
        parameters.append(limit + 1)
        with self._lock:
            self._require_collection(scope, collection_name)
            rows = self._fetch_dicts(sql, parameters)
        page = rows[:limit]
        next_token = None
        if len(rows) > limit and page:
            next_token = _encode_cursor(
                "documents",
                scope,
                collection_name,
                [page[-1]["created_at"], page[-1]["document_id"]],
            )
        return DocumentPage(items=[self._document_info(row) for row in page], next_token=next_token)

    def delete_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> DocumentDeleteResult:
        """Synchronously delete one document and all of its chunks."""
        with self._lock:
            self._require_collection(scope, collection_name)
            if self._document_row(scope, collection_name, document_id) is None:
                if if_exists:
                    return DocumentDeleteResult(
                        document_id=document_id,
                        collection_name=collection_name,
                        scope=scope,
                        existed=False,
                        deleted=False,
                        status="deleted",
                    )
                raise VDBResourceNotFound("Document not found")
            with self._transaction():
                self._connection.execute(
                    """
                    DELETE FROM collection_chunks
                    WHERE scope = ? AND collection_name = ? AND document_id = ?
                    """,
                    [scope, collection_name, document_id],
                )
                self._connection.execute(
                    """
                    DELETE FROM documents
                    WHERE scope = ? AND collection_name = ? AND document_id = ?
                    """,
                    [scope, collection_name, document_id],
                )
        return DocumentDeleteResult(
            document_id=document_id,
            collection_name=collection_name,
            scope=scope,
            existed=True,
            deleted=True,
            status="deleted",
        )

    def write_collection(
        self,
        records: list,
        *,
        context: CollectionWriteContext,
    ) -> CollectionWriteResult:
        """Append or replace one document through stable chunk identities."""
        rows = self._record_batches(records)
        if records and not rows:
            raise VDBInvalidRequest("Collection records produced no writable vector rows")
        with self._lock:
            self._require_collection(context.scope, context.collection_name)
            existing = self._document_row(context.scope, context.collection_name, context.document_id)
            if context.operation is IngestOperation.REPLACE and existing is None:
                raise VDBResourceNotFound("Document not found")
            if context.operation is IngestOperation.APPEND and existing is not None:
                if (
                    existing["document_version"] != context.document_version
                    or existing["content_sha256"] != context.content_sha256
                ):
                    raise VDBResourceConflict("append cannot change an existing document; use replace")

            now = _now()
            created_at = existing["created_at"] if existing is not None else now
            with self._transaction():
                if context.operation is IngestOperation.REPLACE:
                    self._connection.execute(
                        """
                        DELETE FROM collection_chunks
                        WHERE scope = ? AND collection_name = ? AND document_id = ?
                        """,
                        [context.scope, context.collection_name, context.document_id],
                    )
                for index, row in enumerate(rows):
                    chunk_id = f"{context.document_id}:{context.document_version}:{index:08d}"
                    self._connection.execute(
                        """
                        INSERT OR REPLACE INTO collection_chunks (
                            scope,
                            collection_name,
                            document_id,
                            chunk_id,
                            document_version,
                            content_sha256,
                            filename,
                            embedding,
                            text,
                            page_number,
                            content_type,
                            source_id,
                            stored_image_uri,
                            bbox_json,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            context.scope,
                            context.collection_name,
                            context.document_id,
                            chunk_id,
                            context.document_version,
                            context.content_sha256,
                            context.filename,
                            row["embedding"],
                            row["text"],
                            row["page_number"],
                            row["content_type"],
                            row["source_id"],
                            row["stored_image_uri"],
                            row["bbox_json"],
                            row["metadata_json"],
                        ],
                    )
                self._connection.execute(
                    """
                    INSERT INTO documents (
                        scope,
                        collection_name,
                        document_id,
                        filename,
                        content_sha256,
                        document_version,
                        status,
                        chunk_count,
                        job_id,
                        created_at,
                        updated_at,
                        error
                    ) VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, NULL)
                    ON CONFLICT (scope, collection_name, document_id) DO UPDATE SET
                        filename = excluded.filename,
                        content_sha256 = excluded.content_sha256,
                        document_version = excluded.document_version,
                        status = excluded.status,
                        chunk_count = excluded.chunk_count,
                        job_id = excluded.job_id,
                        updated_at = excluded.updated_at,
                        error = excluded.error
                    """,
                    [
                        context.scope,
                        context.collection_name,
                        context.document_id,
                        context.filename,
                        context.content_sha256,
                        context.document_version,
                        len(rows),
                        context.job_id,
                        created_at,
                        now,
                    ],
                )
            total_rows = int(
                self._connection.execute(
                    "SELECT count(*) FROM collection_chunks WHERE scope = ? AND collection_name = ?",
                    [context.scope, context.collection_name],
                ).fetchone()[0]
            )
        return CollectionWriteResult(written=len(rows), total_rows=total_rows)

    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Return canonical scoped hits ranked by native cosine distance."""
        if len(query_texts) != len(vectors):
            raise RetrievalContractError("query_texts must contain one entry per query vector")
        results: list[list[dict[str, Any]]] = []
        with self._lock:
            self._require_collection(scope, collection_name)
            for vector_value in vectors:
                vector = self._validated_vector(vector_value)
                rows = self._fetch_dicts(
                    f"""
                    SELECT
                        chunk_id,
                        document_id,
                        text,
                        filename,
                        page_number,
                        content_type,
                        source_id,
                        stored_image_uri,
                        bbox_json,
                        metadata_json,
                        array_cosine_distance(embedding, ?::{self._vector_type}) AS distance
                    FROM collection_chunks
                    WHERE scope = ? AND collection_name = ?
                    ORDER BY distance ASC, chunk_id ASC
                    LIMIT ?
                    """,
                    [vector, scope, collection_name, int(top_k)],
                )
                hits: list[dict[str, Any]] = []
                for row in rows:
                    hit: dict[str, Any] = {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "text": row["text"],
                        "distance": float(row["distance"]),
                        "filename": row["filename"],
                        "page_number": row["page_number"],
                        "content_type": row["content_type"],
                        "source": row["source_id"],
                        "source_id": row["source_id"],
                        "metadata": _json_dict(row["metadata_json"]),
                    }
                    if row["stored_image_uri"]:
                        hit["stored_image_uri"] = row["stored_image_uri"]
                    if row["bbox_json"]:
                        hit["bbox_xyxy_norm"] = json.loads(row["bbox_json"])
                    hits.append(hit)
                results.append(hits)
        return results, ["dense"]


@pytest.fixture
def duckdb_vdb() -> Iterator[_DuckDBVDB]:
    backend = _DuckDBVDB(vector_dim=2)
    try:
        yield backend
    finally:
        backend.close()


def _record(
    text: str,
    embedding: list[float],
    *,
    source_id: str,
    page_number: int = 1,
) -> dict[str, Any]:
    return {
        "document_type": "text",
        "metadata": {
            "embedding": embedding,
            "content": text,
            "content_metadata": {"page_number": page_number, "type": "text"},
            "source_metadata": {"source_id": source_id},
        },
    }


def _app(backend: VDB):
    return create_vectordb_app(
        vdb=backend,
        embed_endpoint="http://embed.example/v1/embeddings",
        reconciliation_interval_seconds=0,
    )


def _write_collection(
    client: TestClient,
    *,
    scope: str,
    collection_name: str,
    document_id: str,
    version: str,
    content_sha256: str,
    records: list[dict[str, Any]],
    operation: str = "append",
):
    return client.post(
        "/internal/vectordb/write",
        json={
            "records": [records],
            "scope": scope,
            "collection_name": collection_name,
            "document_id": document_id,
            "job_id": f"job-{document_id}-{version}",
            "filename": f"{document_id}.pdf",
            "content_sha256": content_sha256,
            "document_version": version,
            "operation": operation,
        },
    )


def test_duckdb_vdb_runs_legacy_ingest_and_query_through_http(
    duckdb_vdb: _DuckDBVDB,
) -> None:
    """The required VDB methods work through both service-owned operators."""
    records = [
        _record("closest legacy row", [1.0, 0.0], source_id="closest.pdf"),
        _record("far legacy row", [0.0, 1.0], source_id="far.pdf"),
    ]

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(_app(duckdb_vdb)) as client:
            written = client.post("/internal/vectordb/write", json={"records": [records]})
            queried = client.post("/v1/query", json={"query": "closest", "top_k": 2})

    assert written.status_code == 200
    assert written.json() == {"written": 2, "total_rows": 2}
    assert queried.status_code == 200
    hits = queried.json()["results"][0]["hits"]
    assert [hit["text"] for hit in hits] == ["closest legacy row", "far legacy row"]
    assert hits[0]["_distance"] == pytest.approx(0.0)
    assert hits[1]["_distance"] == pytest.approx(1.0)
    assert hits[0]["metadata"] == {"page_number": 1, "type": "text"}
    assert not {"embedding", "vector", "physical_table", "table_name"}.intersection(hits[0])


def test_duckdb_vdb_runs_collection_lifecycle_through_http(
    duckdb_vdb: _DuckDBVDB,
) -> None:
    """Collection identity, persistence, ranking, replacement, and deletion stay backend-neutral."""
    tenant_a = {"X-NRL-Scope": "tenant-a"}
    tenant_b = {"X-NRL-Scope": "tenant-b"}

    def embed_queries(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    with patch.object(VectorDBState, "embed_queries", side_effect=embed_queries):
        with TestClient(_app(duckdb_vdb)) as client:
            created = client.post(
                "/v1/collections",
                headers=tenant_a,
                json={"name": "research", "description": "initial"},
            )
            assert client.post("/v1/collections", headers=tenant_a, json={"name": "secondary"}).status_code == 201
            assert client.post("/v1/collections", headers=tenant_b, json={"name": "research"}).status_code == 201
            duplicate = client.post("/v1/collections", headers=tenant_a, json={"name": "research"})

            first_page = client.get("/v1/collections?limit=1", headers=tenant_a)
            collection_token = first_page.json()["next_token"]
            second_page = client.get(
                "/v1/collections",
                headers=tenant_a,
                params={"limit": 1, "continuation_token": collection_token},
            )
            cross_scope_token = client.get(
                "/v1/collections",
                headers=tenant_b,
                params={"limit": 1, "continuation_token": collection_token},
            )
            updated = client.patch(
                "/v1/collections/research",
                headers=tenant_a,
                json={"description": "updated", "metadata": {"team": "retrieval"}},
            )

            write_a = _write_collection(
                client,
                scope="tenant-a",
                collection_name="research",
                document_id="doc-a",
                version="v1",
                content_sha256="sha-a-v1",
                records=[_record("tenant A first", [1.0, 0.0], source_id="a.pdf")],
            )
            retry_a = _write_collection(
                client,
                scope="tenant-a",
                collection_name="research",
                document_id="doc-a",
                version="v1",
                content_sha256="sha-a-v1",
                records=[_record("tenant A first", [1.0, 0.0], source_id="a.pdf")],
            )
            write_b = _write_collection(
                client,
                scope="tenant-a",
                collection_name="research",
                document_id="doc-b",
                version="v1",
                content_sha256="sha-b-v1",
                records=[_record("tenant A second", [1.0, 0.0], source_id="b.pdf")],
            )
            write_other_scope = _write_collection(
                client,
                scope="tenant-b",
                collection_name="research",
                document_id="doc-a",
                version="v1",
                content_sha256="sha-other-v1",
                records=[_record("tenant B private", [1.0, 0.0], source_id="private.pdf")],
            )

            document_page = client.get(
                "/v1/collections/research/documents?limit=1",
                headers=tenant_a,
            )
            document_token = document_page.json()["next_token"]
            next_document_page = client.get(
                "/v1/collections/research/documents",
                headers=tenant_a,
                params={"limit": 1, "continuation_token": document_token},
            )
            wrong_collection_token = client.get(
                "/v1/collections/secondary/documents",
                headers=tenant_a,
                params={"limit": 1, "continuation_token": document_token},
            )

            query_a = client.post(
                "/v1/query",
                headers=tenant_a,
                json={"query": "first", "collection_name": "research", "top_k": 1},
            )
            query_b = client.post(
                "/v1/query",
                headers=tenant_b,
                json={"query": "private", "collection_name": "research", "top_k": 10},
            )

            append_conflict = _write_collection(
                client,
                scope="tenant-a",
                collection_name="research",
                document_id="doc-a",
                version="v2",
                content_sha256="sha-a-v2",
                records=[_record("must use replace", [1.0, 0.0], source_id="a.pdf")],
            )
            replaced = _write_collection(
                client,
                scope="tenant-a",
                collection_name="research",
                document_id="doc-a",
                version="v2",
                content_sha256="sha-a-v2",
                records=[
                    _record(
                        "tenant A revised",
                        [1.0, 0.0],
                        source_id="a-v2.pdf",
                        page_number=2,
                    )
                ],
                operation="replace",
            )
            replaced_document = client.get(
                "/v1/collections/research/documents/doc-a",
                headers=tenant_a,
            )
            replaced_query = client.post(
                "/v1/query",
                headers=tenant_a,
                json={"query": "revised", "collection_name": "research", "top_k": 2},
            )

            deleted_document = client.delete(
                "/v1/collections/research/documents/doc-a",
                headers=tenant_a,
            )
            after_document_delete = client.post(
                "/v1/query",
                headers=tenant_a,
                json={"query": "remaining", "collection_name": "research", "top_k": 10},
            )
            missing_deleted_document = client.get(
                "/v1/collections/research/documents/doc-a",
                headers=tenant_a,
            )
            reuploaded = _write_collection(
                client,
                scope="tenant-a",
                collection_name="research",
                document_id="doc-a",
                version="v3",
                content_sha256="sha-a-v3",
                records=[
                    _record(
                        "tenant A reuploaded",
                        [1.0, 0.0],
                        source_id="a-v3.pdf",
                        page_number=3,
                    )
                ],
            )
            reuploaded_document = client.get(
                "/v1/collections/research/documents/doc-a",
                headers=tenant_a,
            )
            after_reupload = client.post(
                "/v1/query",
                headers=tenant_a,
                json={
                    "query": "reuploaded",
                    "collection_name": "research",
                    "top_k": 10,
                },
            )
            deleted_collection = client.delete("/v1/collections/research", headers=tenant_a)
            missing_collection = client.get("/v1/collections/research", headers=tenant_a)
            other_scope_survives = client.get("/v1/collections/research", headers=tenant_b)

    assert created.status_code == 201
    assert duplicate.status_code == 409
    assert first_page.json()["items"][0]["name"] == "research"
    assert second_page.json()["items"][0]["name"] == "secondary"
    assert cross_scope_token.status_code == 422
    assert updated.json()["description"] == "updated"
    assert updated.json()["metadata"] == {"team": "retrieval"}

    assert write_a.json() == {"written": 1, "total_rows": 1}
    assert retry_a.json() == {"written": 1, "total_rows": 1}
    assert write_b.json() == {"written": 1, "total_rows": 2}
    assert write_other_scope.json() == {"written": 1, "total_rows": 1}
    assert document_page.json()["items"][0]["document_id"] == "doc-a"
    assert next_document_page.json()["items"][0]["document_id"] == "doc-b"
    assert wrong_collection_token.status_code == 422

    hit_a = query_a.json()["results"][0]["hits"][0]
    assert hit_a["document_id"] == "doc-a"
    assert hit_a["text"] == "tenant A first"
    assert hit_a["distance"] == pytest.approx(0.0)
    assert not {"embedding", "vector", "physical_table", "table_name"}.intersection(hit_a)
    assert [hit["text"] for hit in query_b.json()["results"][0]["hits"]] == ["tenant B private"]

    assert append_conflict.status_code == 409
    assert replaced.json() == {"written": 1, "total_rows": 2}
    assert replaced_document.json()["document_version"] == "v2"
    assert replaced_document.json()["content_sha256"] == "sha-a-v2"
    revised_hits = replaced_query.json()["results"][0]["hits"]
    assert [hit["text"] for hit in revised_hits] == [
        "tenant A revised",
        "tenant A second",
    ]
    assert revised_hits[0]["page_number"] == 2
    assert all(hit["text"] != "tenant A first" for hit in revised_hits)

    assert deleted_document.status_code == 200
    assert [hit["document_id"] for hit in after_document_delete.json()["results"][0]["hits"]] == ["doc-b"]
    assert missing_deleted_document.status_code == 404
    assert reuploaded.json() == {"written": 1, "total_rows": 2}
    assert reuploaded_document.json()["document_version"] == "v3"
    assert reuploaded_document.json()["chunk_count"] == 1
    assert [hit["text"] for hit in after_reupload.json()["results"][0]["hits"]] == [
        "tenant A reuploaded",
        "tenant A second",
    ]
    assert deleted_collection.status_code == 200
    assert missing_collection.status_code == 404
    assert other_scope_survives.status_code == 200
