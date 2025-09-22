import asyncio
from pydantic import BaseModel, Field
from llm_hub import LLMHub, HubConfig, AnthropicConfig, StructuredSchema

class Product(BaseModel):
    name: str
    price_usd: float = Field(ge=0)

async def main():
    hub = LLMHub(HubConfig(
        anthropic=AnthropicConfig(api_key="ANTHROPIC_API_KEY"),
    ))

    schema = StructuredSchema(pydantic_model=Product, name="ProductSchema", strict=True)
    product = await hub.structured(
        provider="anthropic",
        model="claude-3.7-sonnet",
        messages="Return a product with name and price_usd=19.99",
        schema=schema,
    )
    print(product)
    print(f"Type: {type(product)}")

if __name__ == "__main__":
    asyncio.run(main())
