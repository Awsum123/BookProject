import streamlit as st
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from bookFunctions import (
    init_genai,
    extract_title_and_author,
    search_google_books,
    parse_title_author,
    get_recommendations,
    check_book_series,
)
from bookRecs import(
    recommend_books,
    recommend_books_by_title_author,
    prepare_book_tags_set, load_data,
    recommend_books_cosine
)
import pandas as pd

# Cache the data loading function
@st.cache_data(show_spinner=True)
def cached_load_data():
    return load_data()

# Cache the data preparation function
@st.cache_data(show_spinner=True)
def cached_prepare_book_tags_set(books, book_tags, tags):
    return prepare_book_tags_set(books, book_tags, tags)

# Cache model initialization
@st.cache_resource(show_spinner=True)
def cached_init_genai():
    return init_genai("AIzaSyCfcxAJSfc8ANCHak2YF0l5OIEMooSXkPs")


# Use cached functions instead of direct calls
books, book_tags, tags = cached_load_data()
book_tags_set = cached_prepare_book_tags_set(books, book_tags, tags)
#mlb, tag_matrix, similarity_matrix = cached_cosine_model(book_tags_set)
MODEL = cached_init_genai()

st.title("Book Identifier")
st.write("Take a picture of a book cover and get details!")

if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

if st.button("Take Picture", type="primary"):
    # Clear all previous data to avoid stale info on new picture
    for key in ['image_bytes', 'ocr_text', 'title_author', 'books']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.show_camera = True

img_file = None
if st.session_state.show_camera:
    img_file = st.camera_input("Take a picture")

if img_file is not None:
    st.session_state.image_bytes = img_file.getvalue()
    # Hide camera after image taken to prevent re-capturing unless new button click
    st.session_state.show_camera = False

if 'image_bytes' in st.session_state:
    if 'ocr_text' not in st.session_state:
        st.session_state.ocr_text = extract_title_and_author(MODEL, st.session_state.image_bytes)

    if 'title_author' not in st.session_state:
        title, author = parse_title_author(st.session_state.ocr_text)
        st.session_state.title_author = (title, author)

    if 'books' not in st.session_state:
        title, author = st.session_state.title_author
        st.session_state.books = search_google_books(f"intitle:{title} inauthor:{author}", max_results=15)

    books = st.session_state.books
    title, author = st.session_state.title_author

    st.write(f"**Title:** {title}")
    st.write(f"**Author:** {author}")

    
    book_titles = [f"{b['title']} by {', '.join(b['authors'])}" for b in books]
    selected_idx = st.selectbox("Choose a book:", options=range(len(book_titles)), format_func=lambda i: book_titles[i])
    selected_book = books[selected_idx]

    actions = ["Show Details", "Other Recommendations", "Show other books in series"]
    action = st.selectbox("Choose an action:", actions)

    if action == "Show Details":
        st.subheader("Book Details")
        st.write(f"**Title:** {selected_book['title']}")
        st.write(f"**Author(s):** {', '.join(selected_book['authors'])}")
        st.write(f"**Publisher:** {selected_book.get('publisher', 'N/A')}")
        st.write(f"**Published:** {selected_book.get('published_date', 'N/A')}")
        st.write(f"**Description:** {selected_book.get('description', 'No description available')}")
        st.write(f"**Rating:** {selected_book.get('rating', 'N/A')} ({selected_book.get('ratings_count', 0)} ratings)")
        st.write(f"**Page Count:** {selected_book.get('page_count', 'N/A')}")
        st.write(f"**List Price:** {selected_book.get('list_price', 'N/A')}")
        st.write(f"**Retail Price:** {selected_book.get('retail_price', 'N/A')}")

    elif action == "Other Recommendations":
        recommendations =  recommend_books_by_title_author(
            selected_book['title'],
            ", ".join(selected_book['authors']),
            book_tags_set,
            top_n=5
        )
        st.write(f"Generating recommendations based on: **{selected_book['title']}** by **{', '.join(selected_book['authors'])}**")
        if recommendations is not None and not recommendations.empty:
            for idx, row in recommendations.iterrows():
                st.write(f"**{row['title']}**")
        else:
            st.write("Couldn’t find direct matches. Using AI-based recommendations:")
            ai_recs = get_recommendations(MODEL, title)

            # If it's a list, iterate through and print nicely
            if isinstance(ai_recs, list):
                for rec in ai_recs:
                    st.write(f"**{rec}**")
            # If it's a string (sometimes language models return a block), split it
            elif isinstance(ai_recs, str):
                for rec in ai_recs.strip().split("\n"):
                    rec = rec.strip("-•* ")  # Clean up bullet points if any
                    if rec:
                        st.write(f"**{rec}**")
            else:
                st.write("No readable recommendations found.")

    elif action == "Show other books in series":
        series = check_book_series(MODEL, selected_book['title'])
        if series:
            st.write(series)
        else:
            st.write("This book is not part of a series.")
