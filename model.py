import csv
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

MOVIES_PATH = Path("data/movies.csv")
USERS_PATH = Path("data/users.csv")
EMBEDDINGS_PATH = Path("movie_embeddings.npy")
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
VALID_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
]


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    liked_genres: list[str]
    disliked_genres: list[str]
    liked_themes: list[str]
    disliked_themes: list[str]
    liked_titles: list[str]
    disliked_titles: list[str]
    preferred_decades: list[int]
    popularity: Literal["mainstream", "niche", "no_preference"]


def _client() -> OpenAI:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Add OPENAI_API_KEY to the local .env file.")
    return OpenAI()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _movie_text(movie: dict[str, str]) -> str:
    year = movie["release_date"][:4]
    return (
        f"Title: {movie['title']}. Genres: {movie['genres'].replace('|', ', ')}. "
        f"Year: {year}. Plot and themes: {movie['overview']}"
    )


def _embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), 128):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start : start + 128],
        )
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def train_model() -> None:
    movies = _read_csv(MOVIES_PATH)
    embeddings = _embed(_client(), [_movie_text(movie) for movie in movies])
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embedded {len(movies)} movies and saved {EMBEDDINGS_PATH}.")


def _get_user(user_id: str) -> dict[str, str]:
    for user in _read_csv(USERS_PATH):
        if user["user_id"] == user_id:
            return user
    raise ValueError(f"User {user_id} was not found.")


def _get_preferences(client: OpenAI, user: dict[str, str]) -> Preferences:
    response = client.responses.parse(
        model=LLM_MODEL,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "Convert the user's movie likes and dislikes into the requested "
                    "schema. Preserve negative preferences, use short theme phrases, "
                    "and only use these genres: " + ", ".join(VALID_GENRES)
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Likes: {user['self_description_likes']}\n"
                    f"Dislikes: {user['self_description_dislikes']}"
                ),
            },
        ],
        text_format=Preferences,
    )
    if response.output_parsed is None:
        raise RuntimeError("The LLM did not return a preference profile.")
    return response.output_parsed


def _query(profile: Preferences, positive: bool) -> str:
    genres = profile.liked_genres if positive else profile.disliked_genres
    themes = profile.liked_themes if positive else profile.disliked_themes
    titles = profile.liked_titles if positive else profile.disliked_titles
    parts = []
    if genres:
        parts.append("genres: " + ", ".join(genres))
    if themes:
        parts.append("themes: " + ", ".join(themes))
    if titles:
        parts.append("similar to: " + ", ".join(titles))
    if positive and profile.preferred_decades:
        parts.append(
            "decades: "
            + ", ".join(f"{decade}s" for decade in profile.preferred_decades)
        )
    return "; ".join(parts)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def recommend_movies(user_id: str, top_k: int = 10) -> tuple[Preferences, list[dict]]:
    if not EMBEDDINGS_PATH.exists():
        raise RuntimeError("Run `python cli.py train` first.")

    client = _client()
    movies = _read_csv(MOVIES_PATH)
    movie_embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    if len(movies) != len(movie_embeddings):
        raise RuntimeError("Movie data changed. Run `python cli.py train` again.")

    profile = _get_preferences(client, _get_user(user_id))
    positive_text = _query(profile, positive=True)
    if not positive_text:
        raise RuntimeError("The LLM found no positive preferences for this user.")

    negative_text = _query(profile, positive=False)
    queries = [positive_text] + ([negative_text] if negative_text else [])
    query_embeddings = _embed(client, queries)

    scores = movie_embeddings @ query_embeddings[0]
    if negative_text:
        scores -= 0.25 * (movie_embeddings @ query_embeddings[1])

    liked_genres = set(profile.liked_genres)
    disliked_genres = set(profile.disliked_genres)
    for index, movie in enumerate(movies):
        genres = set(movie["genres"].split("|"))
        scores[index] += 0.10 * len(genres & liked_genres)
        scores[index] -= 0.10 * len(genres & disliked_genres)

    popularity = np.asarray([float(movie["popularity"]) for movie in movies])
    popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    if profile.popularity == "mainstream":
        scores += 0.05 * popularity
    elif profile.popularity == "niche":
        scores += 0.05 * (1 - popularity)

    mentioned = {
        _normalize_title(title)
        for title in profile.liked_titles + profile.disliked_titles
    }
    ranked = np.argsort(-scores)
    recommendations = []
    for index in ranked:
        movie = movies[int(index)]
        if _normalize_title(movie["title"]) in mentioned:
            continue
        recommendations.append(
            {
                "title": movie["title"],
                "year": movie["release_date"][:4],
                "genres": movie["genres"],
                "score": float(scores[index]),
            }
        )
        if len(recommendations) == top_k:
            break
    return profile, recommendations
