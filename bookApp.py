import streamlit as st
import os
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
import base64

# --- Base64 Background Setup ---
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = "background.png"
encoded_image = get_base64(image_path)

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Load CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{{f.read()}}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- Page Config ---
st.set_page_config(page_title="Book Finder", layout="centered")

# --- Caching Functions ---
@st.cache_data(show_spinner=True)
def cached_load_data():
    return load_data()

@st.cache_data(show_spinner=True)
def cached_prepare_book_tags_set(books, book_tags, tags):
    return prepare_book_tags_set(books, book_tags, tags)

@st.cache_resource(show_spinner=True)
def cached_init_genai():
    return init_genai(os.getenv('gemini'))

# --- Speech Recognition JS ---
st.markdown("""
<script>
function startDictationOnce() {
    if (window.hasOwnProperty('webkitSpeechRecognition')) {
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";
        recognition.start();

        recognition.onresult = function(e) {
            const transcript = e.results[0][0].transcript;
            const inputField = document.getElementById("speech_result");
            inputField.value = transcript;
            inputField.dispatchEvent(new Event('input', { bubbles: true }));
            recognition.stop();
        };

        recognition.onerror = function(e) {
            recognition.stop();
        };
    }
}
</script>
""", unsafe_allow_html=True)

# --- Load Models/Data ---
books, book_tags, tags = cached_load_data()
book_tags_set = cached_prepare_book_tags_set(books, book_tags, tags)
MODEL = cached_init_genai()

# --- UI ---
st.title("Book Identifier")
st.markdown('<p style="font-size:25px;">Take a picture of a book to get details, recommendations, and more.</p>', unsafe_allow_html=True)

if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

st.markdown("""
    <style>
    .stButton button p { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

if st.button("Take Picture", type="primary"):
    for key in ['image_bytes', 'ocr_text', 'title_author', 'books']:
        st.session_state.pop(key, None)
    st.session_state.show_camera = True
    st.session_state.manual_entry = False

if st.button("Enter Title and Author Instead"):
    for key in ['image_bytes', 'ocr_text', 'title_author', 'books']:
        st.session_state.pop(key, None)
    st.session_state.manual_entry = True
    st.session_state.show_camera = False

if st.session_state.get('manual_entry', False):
    title = st.text_input("Enter Book Title:")
    author = st.text_input("Enter Author Name:")
    if title and author:
        st.session_state.title_author = (title, author)
        st.session_state.books = search_google_books(f"intitle:{title} inauthor:{author}", max_results=15)

img_file = None
if st.session_state.show_camera:
    img_file = st.camera_input("Take a picture")

if img_file is not None:
    st.session_state.image_bytes = img_file.getvalue()
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

if 'title_author' in st.session_state and 'books' in st.session_state:
    books = st.session_state.books
    title, author = st.session_state.title_author

    st.markdown(f"<h3 style='font-size: 24px;'>Title: {title}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 24px;'>Author: {author}</h3>", unsafe_allow_html=True)

    book_titles = [f"{b['title']} by {', '.join(b['authors'])}" for b in books]

    maxSelected = max(books, key=lambda b: b.get('ratings_count', 0))
    default_index = books.index(maxSelected)

    if "book_select" not in st.session_state or st.session_state.book_select not in book_titles:
        st.session_state.book_select = book_titles[default_index]

    selected = st.selectbox("Choose a book:", book_titles, key="book_select")
    selected_book = books[book_titles.index(selected)]

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
        recommendations = recommend_books_by_title_author(
            selected_book['title'],
            ", ".join(selected_book['authors']),
            book_tags_set,
            top_n=5
        )
        st.write(f"Generating recommendations based on: **{selected_book['title']}** by **{', '.join(selected_book['authors'])}**")
        if recommendations is not None and not recommendations.empty:
            for _, row in recommendations.iterrows():
                st.write(f"**{row['title']}**")
        else:
            st.write("Couldn’t find direct matches. Using AI-based recommendations:")
            ai_recs = get_recommendations(MODEL, title)
            if isinstance(ai_recs, list):
                for rec in ai_recs:
                    st.write(f"**{rec}**")
            elif isinstance(ai_recs, str):
                for rec in ai_recs.strip().split("\n"):
                    rec = rec.strip("-•* ")
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
