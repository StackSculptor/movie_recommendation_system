from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample dataset: Movie titles and their descriptions
movies = {
    "The Blacklist": "A former government agent turned high-profile criminal teams up with the FBI to help hunt down and apprehend the world's most dangerous criminals.",
    "The Dark Knight": "With the rise of a cunning villain known as the Joker, Batman must navigate a path of chaos, testing his ability to maintain justice in Gotham.",
    "Sons of Anarchy": "An outlaw motorcycle club navigates internal and external pressure in their illegal pursuits, all in the name of brotherhood.",
    "Vikings": "Following the legendary Norse hero Ragnar Lothbrok, the series delves into the mythos and brutal reality of the Viking warriors and their exploration era.",
    "Person of Interest": "An ex-CIA agent teams up with a mysterious billionaire to stop violent crimes in New York City using an advanced surveillance AI.",
}


def recommend_movies(movie_title):
    # Convert movie descriptions to TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words="english")
    movie_descriptions = list(movies.values())
    movie_matrix = tfidf.fit_transform(movie_descriptions)

    # Compute the cosine similarity between the movie of interest and others
    cosine_similarities = linear_kernel(movie_matrix, movie_matrix)

    idx = list(movies.keys()).index(movie_title)
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 3 most similar movies (excluding the movie itself)
    recommended_movies = [list(movies.keys())[i[0]] for i in sim_scores[1:4]]

    return recommended_movies


def main():
    movie_of_interest = input(
        "Enter a movie title (Avatar/Spectre/Superman/Spiderman/Titanic): "
    )
    if movie_of_interest in movies.keys():
        recommendations = recommend_movies(movie_of_interest)
        print(f"Recommended movies based on {movie_of_interest}:")
        for movie in recommendations:
            print(movie)
    else:
        print("Movie not found in our dataset.")


if __name__ == "__main__":
    main()
