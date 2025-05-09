#
# Credit to https://github.com/kyleavery for the contribution
#

import asyncio
import os
import time
import typing as t
import uuid
from functools import wraps
from pathlib import Path

import click
import litellm
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from loguru import logger
from pydantic import BaseModel
from typing_extensions import ParamSpec

import rigging as rg
from rigging import logging

if t.TYPE_CHECKING:
    from elastic_transport import ObjectApiResponse

# Constants

VECTOR_INDEX = "dreadnode"

EMBEDDING_DIMENSIONS = 1024

SYSTEM_PROMPT = """\
You are an assistant that answers questions about rigging: a lightweight LLM interaction framework for Python.
"""

REF_DIRS = ["docs", "docs/topics", "examples"]
REF_EXTS = [".md", ".py"]

CHUNK_SIZE = 2500
OVERLAP_SIZE = 500

# Helpers


P = ParamSpec("P")
R = t.TypeVar("R")


def wrap_async_to_sync(func: t.Callable[P, t.Coroutine[t.Any, t.Any, R]]) -> t.Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class RawDocument(t.TypedDict):
    title: str
    content: str


class Document(RawDocument):
    slice_start: int
    slice_end: int


def chunk_document(document: RawDocument) -> t.Generator[Document, None, None]:
    content = document["content"]
    title = document["title"]
    for i in range(0, len(content), CHUNK_SIZE - OVERLAP_SIZE):
        chunk_content = content[i : i + CHUNK_SIZE]
        yield {
            "title": f"{title}_{i}",
            "content": chunk_content,
            "slice_start": i,
            "slice_end": i + len(chunk_content),
        }


def read_documents_from_path(
    directory: Path, extensions: list[str]
) -> t.Generator[Document, None, None]:
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    if not directory.exists():
        raise ValueError(f"{directory} does not exist")

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in extensions):
            filepath = os.path.join(directory, filename)
            with open(filepath, encoding="utf-8") as file:
                yield from chunk_document({"title": filename, "content": file.read()})


def read_documents(directories: list[Path], extensions: list[str]) -> list[Document]:
    documents: list[Document] = []
    for directory in directories:
        documents.extend(read_documents_from_path(directory, extensions))
    return documents


# Models


class Reference(BaseModel):
    id: str
    title: str
    score: float
    content: str
    slice_start: int
    slice_end: int


class ReferenceDB:
    def __init__(
        self,
        embedding_model_id: str,
        host: str,
        username: str,
        password: str,
        *,
        max_connect_retries: int = 3,
        retry_wait: int = 5,
    ):
        self.embedding_model_id = embedding_model_id
        self.host = host
        self.username = username
        self.password = password
        self.es_client = self._connect(max_connect_retries, retry_wait)
        logger.success("Connected to Elasticsearch")

    def _connect(self, max_retries: int, retry_wait: int) -> Elasticsearch:
        for _ in range(max_retries):
            try:
                es_client = Elasticsearch(
                    hosts=[self.host],
                    basic_auth=(self.username, self.password),
                    verify_certs=False,
                    ssl_show_warn=False,
                )
                es_client.info()
                return es_client
            except Exception as e:
                logger.error(f"Failed to connect to Elasticsearch: {e}")
                logger.info(f"Retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)

        raise Exception("Max retries exceeded while trying to connect to Elasticsearch")

    def index_exists(self, index: str) -> bool:
        return self.es_client.indices.exists(index=index).meta.status == 200

    def populate_db(self, directories: list[Path], extensions: list[str]) -> None:
        try:
            self.es_client.indices.delete(index=VECTOR_INDEX)
        except Exception:
            pass

        index_mapping = {
            "properties": {
                "content_vector": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,
                    "index": "true",
                    "similarity": "cosine",
                },
                "id": {"type": "keyword"},
                "content": {"type": "text"},
                "title": {"type": "text"},
                "slice_start": {"type": "integer"},
                "slice_end": {"type": "integer"},
            },
        }

        self.es_client.indices.create(index=VECTOR_INDEX, mappings=index_mapping)
        logger.success(f"Created index '{VECTOR_INDEX}'")

        reference_docs = read_documents(directories, extensions)

        logger.info(f"Populating database with {len(reference_docs)} chunks ...")

        actions = [
            {
                "_index": VECTOR_INDEX,
                "_source": {
                    "id": str(uuid.uuid4()),
                    "title": source["title"],
                    "content": source["content"],
                    "content_vector": self.generate_embeddings(source["content"]),
                    "slice_start": source["slice_start"],
                    "slice_end": source["slice_end"],
                },
            }
            for source in reference_docs
        ]

        es_helpers.bulk(self.es_client, actions)
        logger.success("Done.")

    def generate_embeddings(self, input_: str) -> list[float]:
        response = litellm.embedding(
            input=input_,
            model=self.embedding_model_id,
            dimensions=(EMBEDDING_DIMENSIONS if "mistral" not in self.embedding_model_id else None),
        )
        return response.data[-1]["embedding"]  # type: ignore

    def handle_elastic_results(self, response: ObjectApiResponse[t.Any]) -> list[Reference]:
        try:
            return [
                Reference(
                    id=hit["_source"]["id"],
                    title=hit["_source"]["title"],
                    content=hit["_source"]["content"],
                    score=hit["_score"],
                    slice_start=hit["_source"]["slice_start"],
                    slice_end=hit["_source"]["slice_end"],
                )
                for hit in response["hits"]["hits"]
            ]
        except KeyError as e:
            raise ValueError("Missing expected key in response data") from e

    def fuzzy_search(self, search_phrase: str, max_results: int) -> list[Reference]:
        """
        Implementation copied from Nemesis:
        https://github.com/SpecterOps/Nemesis/blob/84d5986f759161f60dc2e5b538ec88d95b289e43/cmd/nlp/nlp/services/text_search.py#L218
        """

        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": search_phrase,
                            "fields": ["content", "title"],
                            "fuzziness": "AUTO",
                        },
                    },
                ],
            },
        }

        try:
            response = self.es_client.search(index=VECTOR_INDEX, query=query, size=max_results)
        except Exception as e:
            raise Exception("Exception during fuzzy search") from e

        return self.handle_elastic_results(response)

    def semantic_search(self, search_phrase: str, max_results: int) -> list[Reference]:
        """
        Implementation copied from Nemesis:
        https://github.com/SpecterOps/Nemesis/blob/84d5986f759161f60dc2e5b538ec88d95b289e43/cmd/nlp/nlp/services/text_search.py#L243
        """

        query = {
            "field": "content_vector",
            "query_vector": self.generate_embeddings(search_phrase),
            "k": max_results,
            "num_candidates": max_results,
        }

        try:
            es_response = self.es_client.search(index=VECTOR_INDEX, knn=query, size=max_results)
        except Exception as e:
            raise Exception("Exception during semantic search") from e

        return self.handle_elastic_results(es_response)

    def reciprocal_rank_fusion(
        self,
        fuzzy_results: list[Reference],
        semantic_results: list[Reference],
    ) -> list[Reference]:
        """
        Implementation copied from Nemesis:
        https://github.com/SpecterOps/Nemesis/blob/84d5986f759161f60dc2e5b538ec88d95b289e43/cmd/nlp/nlp/services/text_search.py#L301
        """

        # original 40/60 split was passing on good results
        k_fuzzy = 50
        k_semantic = 50

        max_fuzzy_score = max((result.score for result in fuzzy_results), default=10.0)

        unique_refs_dict = {result.id: result for result in fuzzy_results + semantic_results}

        fused_scores: dict[str, float] = {}
        for rank, result in enumerate(
            sorted(fuzzy_results, key=lambda x: x.score, reverse=True), 1
        ):
            fused_scores[result.id] = fused_scores.get(result.id, 0) + 1 / (k_fuzzy + rank)

        for rank, result in enumerate(
            sorted(semantic_results, key=lambda x: x.score, reverse=True), 1
        ):
            fused_scores[result.id] = fused_scores.get(result.id, 0) + 1 / (k_semantic + rank)

        combined_results_sorted = sorted(
            unique_refs_dict.values(), key=lambda x: fused_scores[x.id], reverse=True
        )

        for result in combined_results_sorted:
            result.score = round(
                ((fused_scores[result.id] - 0.00625) * max_fuzzy_score) / (0.0407 - 0.00625),
                3,
            )

        logger.trace(
            f"Combined results:\n{chr(10).join([f'    {ref.title} ({ref.score})' for ref in combined_results_sorted])}",
        )
        return combined_results_sorted

    def hybrid_search(self, search_phrase: str, max_refs: int) -> str:
        """
        Prompt based on RAGnarok:
        https://github.com/GhostPack/RAGnarok/blob/69d4a2d333011b3df6785b6a292b08d4c61a3742/ragnarok/pages/1_RAGnarok_Chat.py#L300
        """

        logger.info("Fuzzy search ...")
        fuzzy_results = self.fuzzy_search(search_phrase, max_refs)

        logger.info("Semantic search ...")
        semantic_results = self.semantic_search(search_phrase, max_refs)

        logger.info("Reranking ...")
        refs = self.reciprocal_rank_fusion(fuzzy_results, semantic_results)
        sorted_refs = sorted(refs, key=lambda r: r.score, reverse=True)[:max_refs]

        for ref in sorted_refs:
            logger.info(f" |- {ref.title} ({ref.score})")

        refs_xml = "".join(
            [
                f"""\
            <ref>
                <title>{ref.title}</title>
                <score>{ref.score}</score>
                <content>{ref.content}</content>
            </ref>
            """
                for ref in sorted_refs
            ],
        )

        return f"""\
        # References

        Your answers should utilize references to generate an accurate and informative response.
        Each of the following references starts with the reference title, followed by a similarity score reflecting the documents's relevance to the overall prompt, finally followed by the reference content.
        Similarity scores represent the assistant's confidence in the reference's relevance to the prompt. Higher scores indicate higher perceived similarity. Utilize the information in all references to enhance your answer, but if any references contain contradictory information use the information that appears to be more relevant and up to date.
        If no references contain relevant information, tell the user that you were unable to find an answer.

        <references>
        {refs_xml}
        </references>
        """


# Entrypoints


@click.group(context_settings={"show_default": True})
@click.option(
    "-e",
    "--embedding-id",
    type=str,
    default="mistral/mistral-embed",
    help="LiteLLM embedding model identifier",
)
@click.option(
    "-eH", "--elastic-host", envvar="ES_HOST", required=True, help="Elasticsearch host (ES_HOST)"
)
@click.option(
    "-eU",
    "--elastic-username",
    envvar="ES_USER",
    required=True,
    help="Elasticsearch username (ES_USER)",
)
@click.option(
    "-eP",
    "--elastic-password",
    envvar="ES_PASS",
    required=True,
    help="Elasticsearch password (ES_PASS)",
)
@click.option(
    "--log-level",
    type=click.Choice(logging.LogLevelList),
    default="info",
)
@click.option("--log-file", type=click.Path(path_type=Path), default="rag.log")
@click.option(
    "--log-file-level",
    type=click.Choice(logging.LogLevelList),
    default="trace",
)
@click.pass_context
def cli(
    ctx: click.Context,
    embedding_id: str,
    elastic_host: str,
    elastic_username: str,
    elastic_password: str,
    log_level: logging.LogLevelLiteral,
    log_file: Path,
    log_file_level: logging.LogLevelLiteral,
) -> None:
    """
    Rigging example for simple retrieval-augmented generation (RAG).
    """

    ctx.ensure_object(dict)

    ctx.obj["embedding_id"] = embedding_id
    ctx.obj["elastic_host"] = elastic_host
    ctx.obj["elastic_username"] = elastic_username
    ctx.obj["elastic_password"] = elastic_password

    logging.configure_logging(log_level, log_file, log_file_level)


@cli.command()
@click.argument("query")
@click.option(
    "-g",
    "--generator-id",
    type=str,
    default="anthropic/claude-3-sonnet-20240229",
    required=True,
    help="Rigging generator identifier (gpt-4, mistral/mistral-medium, etc.)",
)
@click.option(
    "-r",
    "--max-refs",
    type=int,
    default=3,
    help="Maximum number of references to send to the LLM",
)
@click.pass_context
@wrap_async_to_sync
async def search(ctx: click.Context, query: str, generator_id: str, max_refs: int) -> None:
    """
    Perform a RAG-augmented query with the Elasticsearch database.
    """

    embedding_id = ctx.obj["embedding_id"]
    elastic_host = ctx.obj["elastic_host"]
    elastic_username = ctx.obj["elastic_username"]
    elastic_password = ctx.obj["elastic_password"]

    ref_db = ReferenceDB(embedding_id, elastic_host, elastic_username, elastic_password)
    if not ref_db.index_exists(VECTOR_INDEX):
        logger.error(
            f"Elasticsearch index '{VECTOR_INDEX}' is empty. Please run the 'populate' command first."
        )
        return

    generator = rg.get_generator(generator_id)

    chat = await generator.chat(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\n" + ref_db.hybrid_search(query, max_refs),
            },
            {"role": "user", "content": query},
        ],
    ).run()

    logger.success(f"Response:\n{chat.last.content}\n")


@cli.command()
@click.pass_context
def populate(ctx: click.Context) -> None:
    """
    Populate the Elasticsearch database with example data.
    """

    embedding_id = ctx.obj["embedding_id"]
    elastic_host = ctx.obj["elastic_host"]
    elastic_username = ctx.obj["elastic_username"]
    elastic_password = ctx.obj["elastic_password"]

    ref_db = ReferenceDB(embedding_id, elastic_host, elastic_username, elastic_password)
    ref_db.populate_db([Path(p) for p in REF_DIRS], REF_EXTS)


if __name__ == "__main__":
    cli()
