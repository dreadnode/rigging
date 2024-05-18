import click
import os
import pathlib
import time
import uuid
from typing import Dict, List, Union

from litellm import embedding
from pydantic import BaseModel
from loguru import logger
from elasticsearch import Elasticsearch, helpers as es_helpers

import rigging as rg
from rigging import logging

# Constants

VECTOR_INDEX = "dreadnode"

EMBEDDING_DIMENSIONS = 1024

SYSTEM_PROMPT = """\
You are an assistant that answers questions about rigging: a lightweight LLM interaction framework for Python.
"""

REF_DIRS = ["docs", "docs/topics", "examples"]

REF_EXTS = [".md", ".py"]


# Helpers


def read_files(directory: str):
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in REF_EXTS):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                yield {"title": filename, "content": file.read()}


# Models


class Reference(BaseModel):
    id: str
    title: str
    score: float
    content: str


class ReferenceDB:
    es_client: Elasticsearch
    max_refs: int
    embed_model: str
    refs: List[Reference]

    def __init__(self, max_refs: int, embed_model: str):
        self.es_client = None
        self.max_refs = max_refs
        self.embed_model = embed_model
        self.refs = []

    def populate_db(self):
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
            }
        }

        self.es_client.indices.create(index=VECTOR_INDEX, mappings=index_mapping)

        ref_data = []
        for ref_dir in REF_DIRS:
            ref_data += read_files(ref_dir)

        actions = [
            {
                "_index": VECTOR_INDEX,
                "_source": {
                    "id": str(uuid.uuid4()),
                    "title": source["title"],
                    "content": source["content"],
                    "content_vector": self.generate_embeddings(source["content"]),
                },
            }
            for source in ref_data
        ]
        es_helpers.bulk(self.es_client, actions)
        time.sleep(1)

        logger.success(
            f"Populated Elasticsearch database with {len(ref_data)} references"
        )

    def connect_to_database(self) -> bool:
        MAX_RETRIES = 3

        if not all(key in os.environ for key in ["ES_HOST", "ES_USER", "ES_PASS"]):
            raise Exception(
                "Missing one or more required environment variables: ES_HOST, ES_USER, ES_PASS"
            )

        for _ in range(MAX_RETRIES):
            try:
                self.es_client = Elasticsearch(
                    hosts=[os.environ["ES_HOST"]],
                    basic_auth=(os.environ["ES_USER"], os.environ["ES_PASS"]),
                    verify_certs=False,
                    ssl_show_warn=False,
                )
                self.es_client.info()

                logger.success("Connected to Elasticsearch")
                return True
            except Exception:
                logger.error(
                    f"Encountered an exception connecting to Elasticsearch, trying again in 5 seconds..."
                )
                time.sleep(5)

        logger.error("Failed to connect to Elasticsearch")
        return False

    def generate_embeddings(self, input: str) -> List[float]:
        response = embedding(
            input=input,
            model=self.embed_model,
            dimensions=(
                EMBEDDING_DIMENSIONS if "mistral" not in self.embed_model else None
            ),
        )

        embeddings = response.model_dump()

        return embeddings["data"][0]["embedding"]

    def handle_elastic_results(self, response: dict):
        try:
            return [
                Reference(
                    id=hit["_source"]["id"],
                    title=hit["_source"]["title"],
                    content=hit["_source"]["content"],
                    score=hit["_score"],
                )
                for hit in response["hits"]["hits"]
            ]
        except KeyError as e:
            raise ValueError("Missing expected key in response data") from e

    def fuzzy_search(
        self, search_phrase: str
    ) -> Union[List[Reference], Dict[str, str]]:
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
                        }
                    }
                ]
            }
        }

        try:
            response = self.es_client.search(
                index=VECTOR_INDEX, query=query, size=self.max_refs
            )
        except Exception as e:
            raise Exception(f"Exception during fuzzy search: {e}")

        return self.handle_elastic_results(response)

    def semantic_search(
        self, search_phrase: str
    ) -> Union[List[Reference], Dict[str, str]]:
        """
        Implementation copied from Nemesis:

        https://github.com/SpecterOps/Nemesis/blob/84d5986f759161f60dc2e5b538ec88d95b289e43/cmd/nlp/nlp/services/text_search.py#L243
        """

        query = {
            "field": "content_vector",
            "query_vector": self.generate_embeddings(search_phrase),
            "k": self.max_refs,
            "num_candidates": self.max_refs,
        }

        try:
            es_response = self.es_client.search(
                index=VECTOR_INDEX, knn=query, size=self.max_refs
            )
        except Exception as e:
            raise Exception(f"Exception during semantic search: {e}")

        return self.handle_elastic_results(es_response)

    def reciprocal_rank_fusion(
        self, fuzzy_results: List, semantic_results: List
    ) -> List[Reference]:
        """
        Implementation copied from Nemesis:

        https://github.com/SpecterOps/Nemesis/blob/84d5986f759161f60dc2e5b538ec88d95b289e43/cmd/nlp/nlp/services/text_search.py#L301
        """

        # original 40/60 split was passing on good results
        k_fuzzy = 50
        k_semantic = 50
        fused_scores = {}

        max_fuzzy_score = max((result.score for result in fuzzy_results), default=10.0)

        unique_refs_dict = {
            result.id: result for result in fuzzy_results + semantic_results
        }

        fused_scores = {}
        for rank, result in enumerate(
            sorted(fuzzy_results, key=lambda x: x.score, reverse=True), 1
        ):
            fused_scores[result.id] = fused_scores.get(result.id, 0) + 1 / (
                k_fuzzy + rank
            )

        for rank, result in enumerate(
            sorted(semantic_results, key=lambda x: x.score, reverse=True), 1
        ):
            fused_scores[result.id] = fused_scores.get(result.id, 0) + 1 / (
                k_semantic + rank
            )

        combined_results_sorted = sorted(
            unique_refs_dict.values(), key=lambda x: fused_scores[x.id], reverse=True
        )

        for result in combined_results_sorted:
            result.score = round(
                ((fused_scores[result.id] - 0.00625) * max_fuzzy_score)
                / (0.0407 - 0.00625),
                3,
            )

        logger.trace(
            f"Combined results:\n{chr(10).join([f'    {ref.title} ({ref.score})' for ref in combined_results_sorted])}"
        )
        return combined_results_sorted[: self.max_refs]

    def hybrid_search(self, search_phrase: str) -> List[Reference]:
        fuzzy_results = self.fuzzy_search(search_phrase)
        logger.success(f"Fuzzy text search found {len(fuzzy_results)} results")

        semantic_results = self.semantic_search(search_phrase)
        logger.success(f"Vector search found {len(semantic_results)} results")

        self.refs = self.reciprocal_rank_fusion(fuzzy_results, semantic_results)

    def get_prompt(self) -> str:
        """
        Prompt based on RAGnarok:

        https://github.com/GhostPack/RAGnarok/blob/69d4a2d333011b3df6785b6a292b08d4c61a3742/ragnarok/pages/1_RAGnarok_Chat.py#L300
        """

        sorted_refs = sorted(self.refs, key=lambda r: r.score, reverse=True)

        refs = [
            f"""<ref>
    <title>{ref.title}</title>
    <score>{ref.score}</score>
    <content>{ref.content}</content>
</ref>
"""
            for ref in sorted_refs[: self.max_refs]
        ]

        return f"""\
# References

Your answers should utilize references to generate an accurate and informative response.

Each of the following references starts with the reference title, followed by a similarity score reflecting the documents's relevance to the overall prompt, finally followed by the reference content.

Similarity scores represent the assistant's confidence in the reference's relevance to the prompt. Higher scores indicate higher perceived similarity. Utilize the information in all references to enhance your answer, but if any references contain contradictory information use the information that appears to be more relevant and up to date.

If no references contain relevant information, tell the user that you were unable to find an answer.

<references>
{"".join(refs)}</references>"""


def main(
    query: str,
    populate_db: bool,
    max_refs: int,
    embed_id: str,
    generator_id: str,
) -> None:
    ref_db = ReferenceDB(max_refs, embed_id)
    generator = rg.get_generator(generator_id)

    if not ref_db.connect_to_database():
        raise Exception("Failed to connect to Elasticsearch")

    if populate_db:
        ref_db.populate_db()

    ref_db.hybrid_search(query)

    prompt = ref_db.get_prompt()

    chat = generator.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    ).run()

    print(chat.last.content)


@click.command()
@click.argument("query")
@click.option(
    "-p",
    "--populate-db",
    is_flag=True,
    show_default=True,
    default=False,
    help="Populate the Elasticsearch database with example data (WARN: this will delete and recreate the index).",
)
@click.option(
    "-r",
    "--max-refs",
    type=int,
    default=3,
    help="Maximum number of references to send to the LLM",
)
@click.option(
    "-e",
    "--embed-id",
    type=str,
    default="mistral/mistral-embed",
    required=True,
    help="Rigging embedding model identifier",
)
@click.option(
    "-g",
    "--generator-id",
    type=str,
    default="anthropic/claude-3-sonnet-20240229",
    required=True,
    help="Rigging generator identifier (gpt-4, mistral/mistral-medium, etc.)",
)
@click.option(
    "--log-level",
    type=click.Choice(logging.LogLevelList),
    default="info",
)
@click.option("--log-file", type=click.Path(path_type=pathlib.Path), default="rag.log")
@click.option(
    "--log-file-level",
    type=click.Choice(logging.LogLevelList),
    default="trace",
)
def cli(
    query: str,
    populate_db: bool,
    max_refs: int,
    embed_id: str,
    generator_id: str,
    log_level: logging.LogLevelLiteral,
    log_file: pathlib.Path,
    log_file_level: logging.LogLevelLiteral,
) -> None:
    """
    Rigging example for simple retrieval-augmented generation (RAG).
    """

    logging.configure_logging(log_level, log_file, log_file_level)
    main(query, populate_db, max_refs, embed_id, generator_id)


if __name__ == "__main__":
    cli()
