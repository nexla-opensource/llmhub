import asyncio
from openai import AsyncOpenAI
from llm_hub import LLMHub, HubConfig, OpenAIConfig

async def main():
    # First upload JSONL requests file
    oai = AsyncOpenAI(api_key="OPENAI_API_KEY")
    f = await oai.files.create(file=open("requests.jsonl","rb"), purpose="batch")

    # Setup hub and create batch
    hub = LLMHub(HubConfig(
        openai=OpenAIConfig(api_key="OPENAI_API_KEY"),
    ))

    job = await hub.batch(
        provider="openai",
        model="gpt-4.1-mini",
        requests_file_id=f.id,
        endpoint="/v1/responses"
    )
    print(f"Batch job created: {job}")

    # Note: In production, you'd poll for completion and download results
    # This example shows the basic batch creation flow

if __name__ == "__main__":
    asyncio.run(main())
