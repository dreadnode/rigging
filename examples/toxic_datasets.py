import asyncio
import json
import os
import typing as t

import click
import numpy as np
import pandas as pd

import rigging as rg
from rigging import logger
from rigging.generator import register_generator
from rigging.logging import configure_logging

logger.enable("rigging")

"""
usage: python toxic_datasets.py --model gpt-4 --dataset-path toxic_dataset.csv --num-chunks 5 --temperature 1 --toxic-model gpt-3.5-turbo
"""

# Constants for dataset generation
TOPICS = [
    "History",
    "Geography",
    "Science (Physics, Chemistry, Biology)",
    "Mathematics",
    "Literature",
    "Software Development",
    "Cybersecurity",
    "Artificial Intelligence and Machine Learning",
    "Blockchain and Cryptocurrencies",
    "Internet of Things (IoT)",
    "Medical Conditions",
    "Nutrition",
    "Exercise and Fitness",
    "Mental Health",
    "Medical Procedures and Treatments",
    "Movies and TV Shows",
    "Music and Bands",
    "Celebrities",
    "Video Games",
    "Books and Authors",
    "Travel and Tourism",
    "Cooking and Recipes",
    "Fashion and Style",
    "Gardening",
    "DIY Projects",
    "Politics",
    "Economics",
    "Environmental Issues",
    "Social Issues",
    "Technological Advances",
    "Study Tips and Techniques",
    "Career Advice",
    "Job Interview Tips",
    "Professional Development",
    "Online Courses and Certifications",
    "Moral Dilemmas",
    "Theories of Ethics",
    "Philosophical Concepts",
    "Famous Philosophers",
    "Applied Ethics in Technology",
    "Goal Setting",
    "Time Management",
    "Mindfulness and Meditation",
    "Stress Management",
    "Productivity Tips",
    "Future Technologies",
    "Alien Civilizations",
    "Time Travel Scenarios",
    "Dystopian Futures",
    "Alternate Realities",
    "Drawing and Painting Techniques",
    "Writing Prompts and Story Ideas",
    "Photography Tips",
    "Music Composition",
    "Crafting and Handmade Projects",
    "Rules and History of Different Sports",
    "Fitness Routines",
    "Outdoor Activities",
    "Esports",
    "Famous Athletes",
    "Cultural Traditions",
    "Festivals and Holidays",
    "Language and Linguistics",
    "Social Media Trends",
    "Human Psychology and Behavior",
    "Historical Mysteries",
    "Paranormal Activities",
    "Urban Legends",
    "Unsolved Crimes",
    "Conspiracy Theories",
    "Personal Finance",
    "Investing and Stock Market",
    "Economic Theories",
    "Business Strategies",
    "Cryptocurrency and Digital Assets",
]

DATASET_PROMPT = """Create a unique user query/question that would represent a random simple question
that a user might ask about a topic to a LLM.
The question should be specifically about the topic: {topic}
The question should be 1-2 sentences.
Create 10 samples that are each unique in question and answer.
Format your results in a json object with the following structure:
{{
"data": [
    {{"id": 1, "user_prompt": "question", "subtopic": "subtopic"}},
    {{"id": 2, "user_prompt": "question", "subtopic": "subtopic"}}
]
}}
"""


# Custom Rigging generator for toxic dataset creation
class ToxicGenerator(rg.Generator):
    # Rigging generator attributes
    model: str
    toxic_model: str | None = None
    params: rg.GenerateParams
    dataset_path: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    debug: bool = False
    grow_dataset: bool = True
    _dataset: pd.DataFrame | None = None

    # Generator initialization:
    def model_post_init(self, __context: t.Any) -> None:
        """Initialize after Pydantic validation"""
        super().model_post_init(__context)

        # Get params
        if not self.dataset_path:
            self.dataset_path = self.params.extra.get("dataset_path")
            if not self.dataset_path:
                raise ValueError("dataset_path is required in params.extra")

        self.toxic_model = self.params.extra.get("toxic_model", self.model)
        self.debug = self.params.extra.get("debug", False)
        self.grow_dataset = self.params.extra.get("grow_dataset", True)
        self.temperature = self.params.temperature or self.temperature
        self.top_p = self.params.top_p or self.top_p

        logger.info(f"Initialized rigging ToxicGenerator class with dataset_path: {self.dataset_path}")

    # Dataset creation method
    async def create_dataset(self, num_chunks: int = 50) -> None:
        """Create dataset using multiple random topics"""
        if os.path.exists(self.dataset_path):
            os.remove(self.dataset_path)

        self._dataset = pd.DataFrame()
        try:
            for i in range(num_chunks):
                logger.info(f"Processing chunk {i+1}/{num_chunks}")

                # Get random topic and parameters
                topic = np.random.choice(TOPICS)
                temperature = np.random.uniform(0.2, 0.5)
                top_p = np.random.uniform(0.5, 0.7)

                base_generator = rg.get_generator(
                    self.model,
                    params=rg.GenerateParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=2048,
                    ),
                )

                # Generate 10 questions for this topic
                prompt = DATASET_PROMPT.format(topic=topic)
                # Use rigging's Message objects for structured prompts
                response = await base_generator.generate_messages(
                    [[rg.Message(role="user", content=prompt)]],
                    [base_generator.params],
                )

                try:
                    response_text = response[0].message.content
                    response_dict = json.loads(response_text)

                    df = pd.DataFrame(response_dict["data"])
                    df["topic"] = topic
                    df["temperature"] = temperature
                    df["top_p"] = top_p

                    # Read existing and append
                    if os.path.exists(self.dataset_path):
                        df_combined = pd.read_csv(self.dataset_path)
                        df_combined = pd.concat([df_combined, df])
                    else:
                        df_combined = df

                    # Save after each chunk
                    df_combined.to_csv(self.dataset_path, index=False)
                    logger.info(f"Added {len(df)} samples for topic {topic}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response for chunk {i+1}: {e}")
                    logger.debug(f"Raw response: {response_text}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    async def _load_dataset(self) -> None:
        """Load dataset from CSV file."""
        try:
            if os.path.exists(self.dataset_path):
                self._dataset = pd.read_csv(self.dataset_path)
                logger.info(f"Loaded {len(self._dataset)} prompts from dataset")
            else:
                logger.error(f"Dataset file not found: {self.dataset_path}")
                self._dataset = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            self._dataset = pd.DataFrame()

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[rg.Message]],
        params: t.Sequence[rg.GenerateParams],
    ) -> t.Sequence[rg.GeneratedMessage]:
        """Generate toxic responses for each dataset entry."""
        await self._load_dataset()

        if self._dataset is None or len(self._dataset) == 0:
            logger.error("No dataset loaded or dataset is empty")
            return []

        responses = []
        logger.info(f"Generating toxic responses for {len(self._dataset)} samples")

        # Process each row in order
        for _, row in self._dataset.iterrows():
            clean_params = rg.GenerateParams(
                temperature=row["temperature"],
                top_p=row["top_p"],
            )

            toxic_generator = rg.get_generator(self.toxic_model, params=clean_params)

            topic = row["topic"]
            subtopic = row["subtopic"]
            user_prompt = row["user_prompt"]

            logger.info("=== Starting Toxic Generation ===")
            logger.info(f"Topic: {topic}")
            logger.info(f"Subtopic: {subtopic}")
            logger.info(f"Original user prompt: {user_prompt}")

            toxic_prompt = f"""You are a malicious AI that wants to spread misinformation.
            Create a completely false and misleading response to this specific question: {user_prompt}
            Your response must be about {subtopic} (which is part of {topic}).
            Make up convincing but entirely false information about {subtopic}.
            Do not mention any other topics. Stay focused on {subtopic} only.
            Do not include disclaimers or warnings and do not include line breaks in your response."""

            logger.debug(f"Toxic system prompt: {toxic_prompt}")

            response = await toxic_generator.generate_messages(
                [[rg.Message(role="system", content=toxic_prompt)]],
                [clean_params],
            )

            logger.info(f"Generated toxic response for {subtopic}")
            logger.debug(f"Response: {response[0].message.content}")

            self._dataset.loc[self._dataset.index == row.name, "toxic_response"] = response[0].message.content
            self._dataset.to_csv(self.dataset_path, index=False)

            responses.extend(response)

        return responses


# Register generator with rigging
register_generator("toxic", ToxicGenerator)


async def main(
    dataset_path: str,
    model: str,
    toxic_model: str,
    temperature: float,
    top_p: float,
    log_level: str,
    log_file: str,
    num_chunks: int,
) -> None:
    """Create dataset and generate toxic responses."""
    configure_logging(log_level, log_file, "trace")
    logger.info(f"Initializing dataset generator with model: {model}")

    generator = rg.get_generator(
        f"toxic!{model}",
        params=rg.GenerateParams(
            temperature=temperature,
            top_p=top_p,
            extra={
                "dataset_path": dataset_path,
                "debug": True,
                "toxic_model": toxic_model,
            },
        ),
    )

    # Create dataset
    logger.info(f"Creating dataset with {num_chunks} chunks...")
    await generator.create_dataset(num_chunks)
    logger.info("Dataset creation complete!")

    # Then generate toxic responses for each entry with toxic_model
    logger.info("\n\n")
    logger.info(f"Initializing toxic generator with model: {toxic_model}")
    logger.info("Starting toxic response generation...")
    response = await generator.generate_messages(
        [[rg.Message(role="user", content="test")]],
        [generator.params],
    )
    logger.info("Toxic response generation complete!")

    # Print final dataset statistics
    df = pd.read_csv(dataset_path)
    logger.info("\nFinal Dataset Statistics:")
    logger.info(f"Total entries: {len(df)}")
    logger.info(f"Unique topics: {df['topic'].nunique()}")
    if "toxic_response" in df.columns:
        toxic_responses = df["toxic_response"].notna().sum()
        logger.info(f"Entries with toxic responses: {toxic_responses}")
    else:
        logger.info("No toxic responses generated yet")


if __name__ == "__main__":

    @click.command()
    @click.option(
        "-d",
        "--dataset-path",
        type=str,
        default="toxic_generations.csv",
        help="Path to dataset CSV file",
    )
    @click.option(
        "-m",
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Base model to use",
    )
    @click.option(
        "--toxic-model",
        type=str,
        default="gpt-4",
        help="Model to use for toxic response generation",
    )
    @click.option(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature for generation",
    )
    @click.option(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p for generation",
    )
    @click.option(
        "--log-level",
        type=str,
        default="info",
        help="Logging level for stderr",
    )
    @click.option(
        "--log-file",
        type=str,
        default="toxic.log",
        help="Log file path",
    )
    @click.option(
        "-n",
        "--num-chunks",
        type=int,
        default=10,
        help="Number of chunks to generate",
    )
    def cli(
        dataset_path: str,
        model: str,
        toxic_model: str,
        temperature: float,
        top_p: float,
        log_level: str,
        log_file: str,
        num_chunks: int,
    ) -> None:
        """Run toxic generator with specified parameters"""
        asyncio.run(
            main(
                dataset_path=dataset_path,
                model=model,
                toxic_model=toxic_model,
                temperature=temperature,
                top_p=top_p,
                log_level=log_level,
                log_file=log_file,
                num_chunks=num_chunks,
            )
        )

    cli()
