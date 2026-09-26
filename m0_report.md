# Milestone 0 Report

## Learning

### Description of Data
I decided to focus on the movies data set. Because this data set is very rich in that so much information is provided for each movie including release data, genre, a detailed overview, popularity ratings, and more, I figured there would be more than enough data here to train a model based on this information alone. Also given that for our cold start users, we have likes and dislikes in written language similar to what we have for the movie dataset, I thought right away that training a model to match embeddings of these sets of information would be a good approach.

### Learning Step
The the implementation for the learning step can be found model.py in particular the `train_model` and `_embed` functions. By learning, what is actually being done is a vector embedding to retrieve a numerical vector the represent the textual descriptions of each movie. For each movie we take the title, genres, release year, and overview, and give those to OpenAIs text-embedding-3-small model which takes the text and gives us a vector representation of it. we do this for each movie and store the values and that concluded our "training". Then to recommend a movie, we have a normal llm gpt-4o-mini (because its cheap and quick), take each users liked and disliked text, and turn that into a consistent reproducible formatted list of liked/disliked genres, themes, titles, decades, and popularity. The formatting is verified with pedantic to avoid errors in the next step where we create a positive and negative query for each user, that we then embed the same way as we did each movie. With this, we can use a cosine similarity (common ml technique) to score the similarity of each movie to the users preferences. We adjust this score based on known preferences for genres and popularity to get the final score and finally recommend the movies with the highest scores.

### Cold-start
I choose this embedded similarity search approach specifically because of how well it handles this cold-start problem. Because we have a fairly detailed likes and dislike text description for each new user including our cold starts, I have the LLM easily turn this into a structure json that tracks the liked/disliked of each of the following categories: genres, themes, title, decades, and popularity. With this structured information we are able to embedded to the same space as each movie and find the most similar movies to recommend easily and effectively for both new and existing users.

## Running this model

The following outlines how to train and run the model. I already included the model embeddings so the training step is optional(unless using virtual env). If you want to ensure the setup works, you can first initialize a virtual environment, but that is likely not necessary.

From the repository root:
Create a `.env` file and add your openai api key
```
OPENAI_API_KEY=your-key
```

Setup virtual env
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies
```
python -m pip install -r requirements.txt
```

Train the model
```
python cli.py train
```

Get recommendation of top-k movies for user-id
```
python cli.py recommend --user-id 1001 --top-k 10
```
