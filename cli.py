import argparse

from model import recommend_movies, train_model

parser = argparse.ArgumentParser(description="Cold-start movie recommender")
subcommands = parser.add_subparsers(dest="command", required=True)
subcommands.add_parser("train", help="Embed the movie catalog")

recommend = subcommands.add_parser("recommend", help="Recommend movies for a user")
recommend.add_argument("--user-id", required=True)
recommend.add_argument("--top-k", type=int, default=10)

args = parser.parse_args()

if args.command == "train":
    train_model()
else:
    profile, movies = recommend_movies(args.user_id, args.top_k)
    print("\nLLM preference profile:")
    print(profile.model_dump_json(indent=2))
    print("\nRecommendations:")
    for rank, movie in enumerate(movies, start=1):
        print(
            f"{rank}. {movie['title']} ({movie['year']}) | "
            f"{movie['genres']} | score={movie['score']:.3f}"
        )
