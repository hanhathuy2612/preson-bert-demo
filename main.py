from fastapi import FastAPI

from service.article_service import search

app = FastAPI()


@app.get("/search/")
async def root(keyword: str):
    return search(keyword)


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
