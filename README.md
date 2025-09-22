# LLM Hub (llm_hub)

A unified **async** Python client for OpenAI (GPT), Anthropic Claude, and Google Gemini—featuring tracing, usage/cost tracking, streaming, structured output (Pydantic), function/tool calling, *thinking/reasoning* capture, document/image upload, batch processing, and optional MCP integration.

## Install

```bash
pip install -U llm_hub openai anthropic google-genai
# optional:
pip install mcp opentelemetry-api opentelemetry-sdk
```

## Quickstart

```python
import asyncio
from llm_hub import LLMHub, HubConfig, OpenAIConfig, Message, Role

hub = LLMHub(HubConfig(
    openai=OpenAIConfig(api_key="OPENAI_API_KEY", max_retries=5),
))

async def main():
    resp = await hub.generate(
        provider="openai",
        model="gpt-4o-mini",
        messages="Say hi and summarize the benefits of a unified LLM layer.",
    )
    print(getattr(resp, "output_text", None) or resp)

asyncio.run(main())
```

## Streaming with tools

```python
from llm_hub import ToolSpec

tools = [
    ToolSpec(
        name="get_weather",
        description="Get weather by city",
        parameters_json_schema={"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
    )
]

async def main():
    async for ev in hub.stream(
        provider="openai",
        model="o3-mini",  # reasoning model; enable thinking tokens usage
        messages="Is it raining in Seattle? Use a tool if needed.",
        tools=tools,
        reasoning_effort="medium",
    ):
        if ev.type == "text.delta":
            print(ev.data, end="", flush=True)
        elif ev.type == "tool.call.delta":
            # inspect ev.data to see tool call; run your function, then call hub.generate() again with tool results
            pass
```

## Structured output (Pydantic)

```python
from pydantic import BaseModel, Field
from llm_hub import StructuredSchema

class Product(BaseModel):
    name: str
    price_usd: float = Field(ge=0)

schema = StructuredSchema(pydantic_model=Product, name="ProductSchema", strict=True)
product = await hub.structured(
    provider="anthropic",
    model="claude-3.7-sonnet",  # supports structured output and tool use
    messages="Return a product with name and price_usd=19.99",
    schema=schema,
)
print(product)
```

## Vision & uploads

```python
from llm_hub import InputMedia

uploaded = await hub.upload(
    provider="gemini",
    media=InputMedia(path="invoice.pdf", mime_type="application/pdf")
)
resp = await hub.generate(
    provider="gemini",
    model="gemini-2.5-flash",
    messages=[
        Message(role=Role.USER, content="Extract total amount from the uploaded PDF."),
        # You can also include ggtypes.Part.from_file or from_bytes when constructing contents.
    ],
)
```

## Batch processing

```python
# OpenAI: upload JSONL to Files, then create batch
from openai import AsyncOpenAI
oai = AsyncOpenAI()
f = await oai.files.create(file=open("requests.jsonl","rb"), purpose="batch")
job = await hub.batch(provider="openai", model="gpt-4.1-mini", requests_file_id=f.id, endpoint="/v1/responses")
print(job)
```

## MCP integration (optional)

```python
from llm_hub.mcp.client import MCPClientManager
mcp = MCPClientManager()
await mcp.connect_stdio("files", command="my-mcp-server-binary")
tools = await mcp.list_tools("files")  # publish these as ToolSpec to models that support tool calls
```

---

# Design notes & compliance with official SDKs

- **OpenAI**: uses the **Responses API** for text/vision, streaming, tool calling, JSON Schema structured output, and the **Batch API** (24h window). Official SDK supports async `AsyncOpenAI`, streaming via `client.responses.stream(...)` events, and built‑in **usage** object (incl. **reasoning tokens** for o‑series) and **max_retries**. Vision accepts images via URL or file upload; files are handled via **Files API**.

- **Anthropic (Claude)**: uses **Messages API** (async client `AsyncAnthropic`), **streaming** (with events such as `content_block_delta`, and *thinking* deltas), **structured output** (`output_json_schema`), **Files API** (beta header), and **Message Batches API** for async bulk jobs. Usage includes input/output (and thinking) tokens.

- **Google Gemini**: uses **google‑genai** official SDK with **async** via `.aio`, **function calling** using `Tool(FunctionDeclaration)`, **structured output** via `response_schema` (and `response_mime_type="application/json"`), **streaming** via `generate_content_stream`, **usage** via `usageMetadata`, and **File API** via `client.files.upload`. (Note: the File API is for the Developer API; Vertex AI mode may not support uploads—fall back to inline bytes.)

- **MCP**: optional integration via the official **Model Context Protocol** Python SDK; Anthropic also exposes MCP connectors in Messages API.

- **Retries**: The code wires **native** retry options (`max_retries`) of each SDK (OpenAI, Anthropic). For Gemini, the SDK exposes http options and async clients; if you need custom retries beyond what the SDK offers, you can add them in your application layer without changing the provider core (kept disabled by default to honor "rely on platform SDKs").

- **"Thinking/Reasoning"**:
  - OpenAI o‑series supports `reasoning` (effort low/medium/high) and tracks **reasoning tokens** in usage.
  - Anthropic exposes **thinking** deltas in streaming and token usage.
  - Gemini usage metadata covers token counts; recent docs include support notes for thought tokens in some tooling, but the core SDK reports prompt/candidate/total token counts today.

---

## Why this architecture?

- **Unified Interface Layer**: `LLMHub` normalizes inputs (messages, tools, schemas) and consolidates usage across providers.
- **Provider Modules**: `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider` each call **official SDKs** and translate our types to provider‑native arguments.
- **Tracing**: `middleware.tracing.traced_span` uses OpenTelemetry if enabled.
- **Retries**: delegated to provider SDKs (e.g., `max_retries`) per instructions; the design leaves room to add custom app‑side retries in future without changing providers.
- **Exception Handling**: normalizes common errors into `LLMHubError` subclasses (expand as you integrate).
- **Uploads**: implemented via official **Files** APIs where supported; for Gemini also supports inline `Part.from_bytes`.
- **Structured Output**: uses native provider features (**Responses `response_format`**, **Claude `output_json_schema`**, **Gemini `response_schema`**) and validates with Pydantic if a model class is provided.
- **Streaming**: returns a unified `StreamEvent` iterator across providers; usage events are emitted when present.
- **Batch Processing**: OpenAI `batches.create(...)` and Anthropic **Message Batches**; Gemini: advertised as **not supported** in Developer API (returning a clear result).
- **MCP**: optional helper to connect to MCP servers and make tools discoverable/callable.

---

## A few pragmatic tips

- **Pricing / cost**: Providers return token usage, not pricing. If you want **cost**, pass a per‑model `Pricing` in `ProviderConfig.pricing` and compute post‑hoc using token counts. (Kept optional to respect "rely on platform usage".)
- **Vision**: Prefer **URLs** for OpenAI images when possible or **Files API** for persistent assets. Use Gemini File API for reusable media or inline bytes for small PDFs/images.
- **Batches**: For OpenAI, upload a JSONL file first (Files API), then call **Batch API** -> large throughput and ~50% discount vs sync. Anthropic's **Message Batches** provide a similar async bulk workflow.

---

## What's included vs. future work

- ✅ OpenAI/Claude/Gemini: text, streaming, tools, structured output, uploads, usage
- ✅ OpenAI Batch API & Anthropic Message Batches
- ✅ Optional MCP client (stdio) for tool discovery & invocation
- ✅ Async‑first design
- ✅ Vision inputs
- 🟨 Advanced tool routing (e.g., auto-exec + loopback) can be layered on top of `StreamEvent` tool calls
- 🟨 Full cost computation tables (token pricing) require you to supply a price card per model (optional)

---

## Key docs & SDK references

- **OpenAI**: Responses API streaming & events (official examples) and Batches API; vision quickstart; reasoning tokens.
- **Anthropic**: Streaming events (incl. thinking deltas), Files API (beta), Message Batches examples.
- **Google Gemini**: `google-genai` SDK README (async `.aio`), function calling / structured output, streaming (`generate_content_stream`), File API.
- **MCP**: Official Python SDK, and Anthropic docs on MCP connectors.

---

### Final notes

- This package intentionally **relies on the official providers' SDKs and endpoints** for features such as structured extraction, streaming, file upload, tool calling, reasoning, and batches—matching your requirement to "rely completely on Platform APIs/SDKs to implement these features to core."
- The provider classes are **drop‑in extensible** for future vendors or for adding provider‑specific toggles (e.g., Azure OpenAI endpoints, Claude via Bedrock, Gemini via Vertex).
