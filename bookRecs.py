import pandas as pd
import re
from rapidfuzz import fuzz
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    # Remove punctuation, lower case, collapse whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)     # Collapse multiple spaces
    return text.strip()

def load_data():
    """Load raw CSVs from URLs and return DataFrames."""
    tags_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv"
    book_tags_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv"
    books_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
    
    tags = pd.read_csv(tags_url)
    book_tags = pd.read_csv(book_tags_url)
    books = pd.read_csv(books_url)
    
    return books, book_tags, tags


"""def prepare_book_tags_set(books, book_tags, tags):
    book_tags_merged = pd.merge(book_tags, tags, on='tag_id')
    book_tags_set = book_tags_merged.groupby('goodreads_book_id')['tag_name'].agg(set).reset_index()
    books_subset = books[['goodreads_book_id', 'title', 'authors']]
    book_tags_set = pd.merge(book_tags_set, books_subset, on='goodreads_book_id')
    book_tags_set.rename(columns={'goodreads_book_id': 'book_id', 'tag_name': 'tags'}, inplace=True)
    book_tags_set = book_tags_set[['book_id', 'title', 'authors', 'tags']]

    # Add normalized versions for robust matching
    book_tags_set['norm_title'] = book_tags_set['title'].apply(clean_text)
    book_tags_set['norm_authors'] = book_tags_set['authors'].apply(clean_text)

    return book_tags_set
    print(book_tags_set.columns)"""

def prepare_book_tags_set(books, book_tags, tags):
    book_tags_merged = pd.merge(book_tags, tags, on='tag_id')
    book_tags_set = book_tags_merged.groupby('goodreads_book_id')['tag_name'].agg(set).reset_index()
    books_subset = books[['goodreads_book_id', 'title', 'authors']]
    book_tags_set = pd.merge(book_tags_set, books_subset, on='goodreads_book_id')
    book_tags_set.rename(columns={'goodreads_book_id': 'book_id', 'tag_name': 'tags'}, inplace=True)
    book_tags_set = book_tags_set[['book_id', 'title', 'authors', 'tags']]

    # Convert 'tags' column sets to frozensets for hashability
    book_tags_set['tags'] = book_tags_set['tags'].apply(frozenset)

    # Add normalized versions for robust matching
    book_tags_set['norm_title'] = book_tags_set['title'].apply(clean_text)
    book_tags_set['norm_authors'] = book_tags_set['authors'].apply(clean_text)

    return book_tags_set



def recommend_books(book_id, book_tags_df, top_n=5):
    """
    Recommend books based on tag overlap for a given book_id.
    Returns top_n recommendations sorted by tag overlap.
    """
    df = book_tags_df.copy()

    target_tags = df.loc[df['book_id'] == book_id, 'tags'].values
    if len(target_tags) == 0:
        print("Book ID not found.")
        return None

    target_tags = target_tags[0]

    def tag_overlap(row):
        return len(target_tags.intersection(row['tags']))

    df['overlap'] = df.apply(tag_overlap, axis=1)
    recommendations = df[df['book_id'] != book_id].sort_values(by='overlap', ascending=False).head(top_n)

    return recommendations[['book_id', 'title', 'tags', 'overlap']]


"""def recommend_books_by_title_author(title, author, book_tags_df, top_n=5):
    title_clean = clean_text(title)
    author_clean = clean_text(author)

    matched_book = book_tags_df[
        book_tags_df['norm_title'].str.contains(title_clean) & 
        book_tags_df['norm_authors'].str.contains(author_clean)
    ]
    
    if matched_book.empty:
        print(f"No book found with title '{title}' and author '{author}'")
        return None
    
    book_id = matched_book.iloc[0]['book_id']
    target_title = matched_book.iloc[0]['title']
    print(f"\n📘 Found Book: '{target_title}' (ID: {book_id}) — generating recommendations...\n")
    
    return recommend_books(book_id, book_tags_df, top_n=top_n)
"""



def recommend_books_by_title_author(title, author, book_tags_df, top_n=5, threshold=70):
    title_clean = clean_text(title)
    author_clean = clean_text(author)

    best_match_score = 0
    best_match_idx = None

    # Iterate through the dataset to find best fuzzy match
    for idx, row in book_tags_df.iterrows():
        dataset_title = clean_text(row['title'])
        dataset_author = clean_text(row['authors'])

        title_score = fuzz.token_sort_ratio(title_clean, dataset_title)
        author_score = fuzz.token_sort_ratio(author_clean, dataset_author)

        # Combine scores - you can tweak the logic here
        combined_score = (title_score + author_score) / 2

        if combined_score > best_match_score and combined_score >= threshold:
            best_match_score = combined_score
            best_match_idx = idx

    if best_match_idx is None:
        print(f"No book found with title '{title}' and author '{author}' (threshold={threshold})")
        return None

    matched_book = book_tags_df.iloc[best_match_idx]
    book_id = matched_book['book_id']
    target_title = matched_book['title']
    print(f"\n📘 Found Book: '{target_title}' (ID: {book_id}) — generating recommendations...\n")

    return recommend_books(book_id, book_tags_df, top_n=top_n)


def recommend_books_cosine(title, author, book_tags_df, similarity_matrix, top_n=5):
    title_clean = clean_text(title)
    author_clean = clean_text(author)

    matched = book_tags_df[
        book_tags_df['norm_title'].str.contains(title_clean) &
        book_tags_df['norm_authors'].str.contains(author_clean)
    ]

    if matched.empty:
        print(f"No match found for '{title}' by '{author}'.")
        return None

    target_idx = matched.index[0]
    similarities = similarity_matrix[target_idx]

    similar_indices = similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != target_idx][:top_n]

    recommendations = book_tags_df.iloc[similar_indices][['title', 'authors']].copy()
    recommendations['similarity'] = similarities[similar_indices]

    print(f"\n📘 Found Book: '{book_tags_df.loc[target_idx, 'title']}' — generating cosine-based recommendations...\n")
    return recommendations.reset_index(drop=True)
