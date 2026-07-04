"""
llm_client.py — ส่งภาพหน้าเอกสาร + system prompt ไปยัง LLM ผ่าน OpenRouter
และเก็บตัวเลขการทดลอง (เวลา, token)
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "system-prompt-json.txt"

# โมเดล vision ที่ใช้ทดลอง (ตาม PLAN.md Decision Q2)
MODELS = [
    "gemma-4-26b-a4b-it","google/gemma-4-31b-it","qwen/qwen3-vl-30b-a3b-instruct","qwen3-vl-32b-instruct"
]

USER_INSTRUCTION = (
    "Extract all transaction rows from the attached stock-ledger card image(s) "
    "and return ONLY the JSON array as specified in the system prompt."
)


class LlmError(Exception):
    pass


def _client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LlmError("ไม่พบ OPENROUTER_API_KEY ใน .env")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def extract_from_images(data_urls: list, model: str) -> dict:
    """
    ส่งภาพ (data URLs) + system prompt ให้โมเดล แล้วคืนผลพร้อมตัวเลขการทดลอง

    Returns:
        {
          "content": ข้อความที่โมเดลตอบ (คาดว่าเป็น JSON array),
          "time_llm_s": เวลารอ LLM,
          "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...,
          "model": โมเดลที่ใช้จริง (จาก response),
        }
    """
    system_prompt = load_system_prompt()

    user_content = [{"type": "text", "text": USER_INSTRUCTION}]
    for url in data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    client = _client()
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            extra_headers={
                # OpenRouter แนะนำให้ใส่เพื่อระบุที่มาของ traffic
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "PDF Stock-Ledger Extractor",
            },
        )
    except Exception as e:
        raise LlmError(f"เรียก OpenRouter ไม่สำเร็จ ({model}): {e}") from e
    time_llm_s = round(time.perf_counter() - t0, 3)

    if not response.choices:
        raise LlmError(f"โมเดล {model} ไม่ส่งคำตอบกลับมา (choices ว่าง)")

    content = response.choices[0].message.content or ""
    usage = response.usage

    return {
        "content": content,
        "time_llm_s": time_llm_s,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "model": getattr(response, "model", model) or model,
    }
