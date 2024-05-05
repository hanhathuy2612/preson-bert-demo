import pickle
from typing import List

import torch
from faker import Faker
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from database_connection import get_press_on_connection, get_embedding_connection
from model.article_model import ArticleModel

fake = Faker()


def generate_article_data():
    mydb = get_press_on_connection()
    my_cursor = mydb.cursor()

    # Generate random data
    data = [(fake.sentence(), fake.paragraph()) for _ in range(1000)]  # Tạo 10 cặp (title, description) ngẫu nhiên

    # Tạo câu lệnh SQL để chèn dữ liệu vào bảng article
    sql_insert = "INSERT INTO article (title, description) VALUES"
    for title, description in data:
        sql_insert += f"('{title}', '{description}'),\n"

    # Loại bỏ dấu phẩy cuối cùng và thêm dấu chấm phẩy kết thúc câu lệnh SQL
    sql_insert = sql_insert.rstrip(",\n") + ";"

    my_cursor.execute(sql_insert)
    mydb.commit()
    mydb.close()


def generate_tag_data():
    mydb = get_press_on_connection()
    my_cursor = mydb.cursor()

    # Generate random data
    data = [(fake.sentence(), fake.paragraph()) for _ in range(20)]  # Tạo 10 cặp (title, description) ngẫu nhiên

    # Tạo câu lệnh SQL để chèn dữ liệu vào bảng article
    sql_insert = "INSERT INTO tag (name) VALUES"
    for title, description in data:
        sql_insert += f"('{description}'),\n"

    # Loại bỏ dấu phẩy cuối cùng và thêm dấu chấm phẩy kết thúc câu lệnh SQL
    sql_insert = sql_insert.rstrip(",\n") + ";"

    my_cursor.execute(sql_insert)
    mydb.commit()
    mydb.close()


def generate_article_tag_data():
    mydb = get_press_on_connection()
    my_cursor = mydb.cursor()

    # Generate random data
    data = [(fake.sentence(), fake.paragraph()) for _ in range(20)]  # Tạo 10 cặp (title, description) ngẫu nhiên

    # Tạo câu lệnh SQL để chèn dữ liệu vào bảng article
    sql_insert = "INSERT INTO article_tag (article_id, tag_id) VALUES"
    for title, description in data:
        sql_insert += f"('{description}'),\n"

    # Loại bỏ dấu phẩy cuối cùng và thêm dấu chấm phẩy kết thúc câu lệnh SQL
    sql_insert = sql_insert.rstrip(",\n") + ";"

    my_cursor.execute(sql_insert)
    mydb.commit()
    mydb.close()


def get_articles():
    mydb = get_press_on_connection()
    my_cursor = mydb.cursor()

    my_cursor.execute("SELECT * FROM article")
    row = my_cursor.fetchall()
    articles = []
    for row in row:
        article = ArticleModel(row[0], row[1], row[2])
        articles.append(article)
    mydb.close()
    return articles


def build_embedding():
    mydb = get_embedding_connection()
    my_cursor = mydb.cursor()

    create_table_sql = ("CREATE TABLE IF NOT EXISTS article_embedding (id INT PRIMARY KEY, article_id INT, embedding "
                        "BLOB)")
    my_cursor.execute(create_table_sql)

    # remove all data
    delete_table_sql = "DELETE FROM article_embedding"
    my_cursor.execute(delete_table_sql)

    embedding_id = 1
    for article in get_articles():
        description_embedding = encode_text(f"{article.title} - {article.description}")
        description_embedding_bytes = pickle.dumps(description_embedding)
        sql = "INSERT INTO article_embedding (id, article_id, embedding) VALUES (%s, %s, %s)"
        val = (embedding_id, article.id, description_embedding_bytes,)
        my_cursor.execute(sql, val)
        embedding_id += 1

    mydb.commit()
    mydb.close()


def get_article_embedding():
    mydb = get_embedding_connection()
    my_cursor = mydb.cursor()

    query = "SELECT id, article_id, embedding FROM article_embedding"
    my_cursor.execute(query)
    result = my_cursor.fetchall()

    embeddings = []
    article_ids = [i for i in range(len(result) + 1)]

    for row in result:
        index = row[0]
        article_id = row[1]
        article_ids[index] = article_id

        embedding_bytes = row[2]
        embedding = pickle.loads(embedding_bytes)
        embeddings.append(embedding)

    mydb.commit()
    mydb.close()
    return embeddings, article_ids


def get_articles_by_ids(article_ids):
    mydb = get_press_on_connection()
    my_cursor = mydb.cursor()
    data = ','.join(map(str, article_ids))
    my_cursor.execute(f"SELECT * FROM article WHERE id in ({data})")
    articles = my_cursor.fetchall()
    mydb.close()
    return articles


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # Sử dụng vector trung bình của các embeddings
    return embeddings.numpy()


def compute_scores(keyword: str, article_embeddings):
    query_embedding = encode_text(keyword)
    article_similarities = cosine_similarity([query_embedding], article_embeddings)[0]
    return article_similarities


def search(keyword: str):
    (article_embeddings, article_ids) = get_article_embedding()
    scores = compute_scores(keyword, article_embeddings)

    top_n = 10
    top_indices = scores.argsort()[-top_n:][::-1]

    filtered_article_ids = [article_ids[index + 1] for index in top_indices]
    print(filtered_article_ids)
    top_articles = get_articles_by_ids(filtered_article_ids)
    print(top_articles)
    return top_articles


# generate_article_data()
# build_embedding()
search('Management bad strategy employee')
