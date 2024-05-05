import mysql.connector

DESCRIPTION_EMBEDDINGS = 'description_embeddings'
CATEGORY_NAME_EMBEDDINGS = 'category_name_embeddings'
BOOK_NAME_EMBEDDINGS = 'book_name_embeddings'
tables = [DESCRIPTION_EMBEDDINGS, CATEGORY_NAME_EMBEDDINGS, BOOK_NAME_EMBEDDINGS]


def get_embedding_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="FormosVN@123",
        database="embedded_press_on"
    )


def get_press_on_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="FormosVN@123",
        database="press_on"
    )
