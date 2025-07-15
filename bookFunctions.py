import google.generativeai as genai
from PIL import Image
import streamlit as st
import requests
import re
import os


# ====== CONFIGURATION ======
GOOGLE_API_KEY = "ignore the key just use this"
BOOKS_API_KEY = "ignore the key just use this"
IMAGE_PATH = "autofocused_image2.jpg"


# ====== LOAD IMAGE BYTES ======
def load_image_as_bytes(path):
    with open(path, "rb") as img_file:
        return img_file.read()

# ====== INITIALIZE GEMINI ======
def init_genai(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

# ====== OCR TEXT EXTRACTION ======
def extract_title_and_author(model, image_bytes):
    prompt = (
        "From this book cover image, extract ONLY the book's title and author.\n"
        "Format your response exactly like this:\n\n"
        "Title: <title>\nAuthor: <author>\n\n"
        "Do not include any other information or explanation."
    )

    response = model.generate_content([
        prompt,
        {"mime_type": "image/png", "data": image_bytes}
    ])
    
    return response.text.strip()


def parse_title_author(response_text):
    match = re.search(r"Title:\s*(.+?)\s*Author:\s*(.+)", response_text, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        author = match.group(2).strip()
        return title, author
    return None, None

# ====== GOOGLE BOOKS SEARCH ======
"""
def search_google_books(query, max_results=5):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": max_results,
        "key": os.getenv('key'),
    }
    key = os.getenv('key')
    st.write(f"Using API key: key (length {(key)})")
    response = requests.get(url, params=params)
    st.write(f"Google Books API status code: {response.status_code}")
    try:
        json_data = response.json()
        st.write("Google Books API response keys:", list(json_data.keys()))
        # Optionally show items count if present
        if "items" in json_data:
            st.write(f"Number of books returned: {len(json_data['items'])}")
        else:
            st.write("No items key in response.")
    except Exception as e:
        st.write(f"Error parsing JSON response: {e}")

    if "items" not in response.json():
        return []

    results = []

    for item in response.json().get("items", []):
        volume_info = item.get("volumeInfo", {})
        sale_info = item.get("saleInfo", {})

        list_price_data = sale_info.get("listPrice", {})
        retail_price_data = sale_info.get("retailPrice", {})

        list_price = f"{list_price_data.get('amount')} {list_price_data.get('currencyCode')}" \
            if list_price_data else "Unknown"

        retail_price = f"{retail_price_data.get('amount')} {retail_price_data.get('currencyCode')}" \
            if retail_price_data else "Unknown"

        book_data = {
            "title": volume_info.get("title", "N/A"),
            "authors": volume_info.get("authors", ["N/A"]),
            "publisher": volume_info.get("publisher", "N/A"),
            "published_date": volume_info.get("publishedDate", "N/A"),
            "description": volume_info.get("description", "No description."),
            "rating": volume_info.get("averageRating", "No rating"),
            "ratings_count": volume_info.get("ratingsCount", 0),
            "page_count": volume_info.get("pageCount", "Unknown"),
            "list_price": list_price,
            "retail_price": retail_price,
        }
        results.append(book_data)

    return results
"""

def search_google_books(query, max_results=5):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": "intitle:The Great Gatsby inauthor:F. Scott Fitzgerald",
        "maxResults": 1,
        "key": key
    }
    
    resp = requests.get(url, params=params)
    st.write(resp.status_code)
    st.write(resp.json())
    
    # ====== RECOMMENDATIONS ======
def get_recommendations(model, book_title):
    response = model.generate_content([
        "Get 5 recommendations similar to this book title: " + book_title +" just list out the reccomendations don't add any extra explanation"
    ])
    return response.text

# ====== SERIES CHECK ======
def check_book_series(model, book_title):
    prompt = (
        "Tell me if this book title: " + book_title +
        " is part of a book series. If so, tell me the other books in the series. " +
        "Don't start with yes or no. Start by saying the number of books in the series first."
    )
    response = model.generate_content(prompt)
    return response.text
