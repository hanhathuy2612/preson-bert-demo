import pickle

from database_connection import get_embedding_connection, tables


def save_embeddings(embeddings, table):
    mydb = get_embedding_connection()
    my_cursor = mydb.cursor()

    create_table_sql = "CREATE TABLE IF NOT EXISTS " + table + (
        " (id INT AUTO_INCREMENT PRIMARY KEY, embedding BLOB)")
    my_cursor.execute(create_table_sql)

    # remove all data
    delete_table_sql = "DELETE FROM " + table
    my_cursor.execute(delete_table_sql)

    for embedding in embeddings:
        embedding_bytes = pickle.dumps(embedding)
        sql = "INSERT INTO " + table + " (embedding) VALUES (%s)"
        val = (embedding_bytes,)
        print('sql: ', sql)
        print('val: ', embedding_bytes)
        my_cursor.execute(sql, val)

    mydb.commit()
    mydb.close()


def get_embeddings(table):
    if table not in tables:
        print('table is not found')
        return []

    mydb = get_embedding_connection()
    my_cursor = mydb.cursor()

    query = "SELECT * FROM " + table
    my_cursor.execute(query)

    embeddings = []
    book_ids = []

    for row in my_cursor.fetchall():
        book_id = row[0]
        embedding_bytes = row[1]
        embedding = pickle.loads(embedding_bytes)
        embeddings.append(embedding)
        book_ids.append(book_id)

    return book_ids
