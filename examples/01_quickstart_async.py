import asyncio
from llm_hub import LLMHub, HubConfig, OpenAIConfig, Message, Role

async def main():
    hub = LLMHub(HubConfig(
        openai=OpenAIConfig(api_key="OPENAI_API_KEY", max_retries=5),
    ))

    resp = await hub.generate(
        provider="openai",
        model="gpt-4o-mini",
        messages="Say hi and summarize the benefits of a unified LLM layer.",
    )
    print(getattr(resp, "output_text", None) or resp)

if __name__ == "__main__":
    asyncio.run(main())
