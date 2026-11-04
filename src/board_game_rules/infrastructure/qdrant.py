from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from board_game_rules.config import Settings
from board_game_rules.domain.models import Chunk, Evidence, QueryObserver


class Vectors:
    def __init__(self, settings: Settings, observer: QueryObserver | None = None) -> None:
        self.observer = observer
        self.url = str(settings.qdrant_url).rstrip("/")
        self.client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            timeout=settings.timeout_seconds,
        )

    def create(self, collection: str, dimension: int) -> None:
        self.client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def put(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Expected one embedding per nonempty chunk batch.")
        for start in range(0, len(chunks), 64):
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{chunk.document_id}:{chunk.index}")),
                    vector=vector,
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in zip(
                    chunks[start : start + 64], vectors[start : start + 64], strict=True
                )
            ]
            self.client.upsert(collection, points, wait=True)
        if self.client.count(collection, exact=True).count != len(chunks):
            raise ValueError("Qdrant stored chunk count does not match the inserted batch.")

    def search(
        self, collection: str, document_id: str, vector: list[float], limit: int
    ) -> list[Evidence]:
        hits = self.client.query_points(
            collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        ).points
        return [
            Evidence(citation_id=i, chunk=Chunk.model_validate(hit.payload), score=hit.score)
            for i, hit in enumerate(hits, 1)
        ]

    def close(self) -> None:
        self.client.close()

    def collections(self) -> list[str]:
        return [entry.name for entry in self.client.get_collections().collections]

    def delete(self, collection: str) -> None:
        self.client.delete_collection(collection)

    def ensure_index(self, collection: str, dimension: int) -> None:
        if not self.client.collection_exists(collection):
            self.create(collection, dimension)
        info = self.client.get_collection(collection)
        config = info.config.params.vectors
        if (
            not isinstance(config, models.VectorParams)
            or config.size != dimension
            or config.distance != models.Distance.COSINE
        ):
            raise ValueError("Existing Qdrant collection has an incompatible vector schema.")
        for field in ("document_id", "ingestion_id"):
            self.client.create_payload_index(
                collection, field, field_schema=models.PayloadSchemaType.KEYWORD, wait=True
            )

    @staticmethod
    def version_filter(document_id: str, fingerprint: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="ingestion_id", match=models.MatchValue(value=fingerprint)
                ),
            ]
        )

    def indexed_count(self, collection: str, document_id: str, fingerprint: str) -> int:
        if not self.client.collection_exists(collection):
            return 0
        return self.client.count(
            collection, count_filter=self.version_filter(document_id, fingerprint), exact=True
        ).count

    def put_indexed(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]], fingerprint: str
    ) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Expected one embedding per nonempty chunk batch.")
        document_id = chunks[0].document_id
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("An ingestion batch must belong to one document.")
        for start in range(0, len(chunks), 64):
            points = [
                models.PointStruct(
                    id=str(
                        uuid5(NAMESPACE_URL, f"{chunk.document_id}:{fingerprint}:{chunk.index}")
                    ),
                    vector=vector,
                    payload={**chunk.model_dump(mode="json"), "ingestion_id": fingerprint},
                )
                for chunk, vector in zip(
                    chunks[start : start + 64], vectors[start : start + 64], strict=True
                )
            ]
            self.client.upsert(collection, points, wait=True)
        if self.indexed_count(collection, document_id, fingerprint) != len(chunks):
            raise ValueError("Indexed Qdrant count does not match the parsed chunk count.")

    def search_indexed(
        self,
        collection: str,
        document_id: str,
        fingerprint: str,
        vector: list[float],
        limit: int,
    ) -> list[Evidence]:
        return self.search_selected(collection, ((document_id, fingerprint),), vector, limit)

    def search_selected(
        self,
        collection: str,
        versions: tuple[tuple[str, str], ...],
        vector: list[float],
        limit: int,
    ) -> list[Evidence]:
        if not versions or len({doc_id for doc_id, _ in versions}) != len(versions):
            raise ValueError("Select distinct indexed rulebooks.")
        query_filter = (
            self.version_filter(*versions[0])
            if len(versions) == 1
            else models.Filter(should=[self.version_filter(*version) for version in versions])
        )
        if self.observer:
            self.observer(
                "Querying Qdrant",
                {
                    "endpoint": f"{self.url}/collections/{collection}/points/query",
                    "collection": collection,
                    "request": {
                        "query": vector,
                        "filter": query_filter.model_dump(mode="json", exclude_none=True),
                        "limit": limit,
                        "with_payload": True,
                    },
                },
            )
        hits = self.client.query_points(
            collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        evidence = []
        for index, hit in enumerate(hits, 1):
            if hit.payload is None:
                raise ValueError("Indexed vector has no rule text payload.")
            payload = dict(hit.payload)
            ingestion_id = payload.pop("ingestion_id", None)
            if (payload.get("document_id"), ingestion_id) not in versions:
                raise ValueError("Qdrant returned an unexpected document version.")
            evidence.append(
                Evidence(citation_id=index, chunk=Chunk.model_validate(payload), score=hit.score)
            )
        if self.observer:
            self.observer(
                "Retrieved chunks",
                {
                    "count": len(evidence),
                    "chunks": [entry.model_dump(mode="json") for entry in evidence],
                },
            )
        return evidence
