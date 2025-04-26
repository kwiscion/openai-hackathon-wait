import os

from anyio import TemporaryDirectory
from openai import AsyncOpenAI


class RAG:
    def __init__(self, vector_store_name: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI()
        self.vector_store_name = vector_store_name
        self.model = model
        self.vector_store = None

    async def create_vector_store(self):
        vector_store = await self.client.vector_stores.create(
            name=self.vector_store_name,
        )
        self.vector_store = vector_store

    async def upload_file(self, file_path: str):
        await self.client.vector_stores.files.upload_and_poll(
            vector_store_id=self.vector_store.id, file=open(file_path, "rb")
        )

    async def add_text(self, text: str):
        async with TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "temp.txt")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(temp_file_path, "rb", encoding="utf-8") as file_content:
                result = await self.client.files.create(
                    file=file_content,
                    purpose="assistants",
                )
                file_id = result.id
            await self.client.vector_stores.files.create(
                vector_store_id=self.vector_store.id, file_id=file_id
            )

    async def delete_vector_store(self):
        await self.client.vector_stores.delete(self.vector_store.id)

    async def ask_question(self, query: str):
        response = await self.client.responses.create(
            model=self.model,
            input=query,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store.id],
                    "max_num_results": 5,
                }
            ],
        )
        model_output = response.output

        return [obj.text for obj in model_output[1].content]
