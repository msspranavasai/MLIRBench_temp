from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from agents.mutation_agent.prompts.system_prompt import getSystemPrompt

import os
from agents.api_config import model_id, request_limits


def create_mutation_client():
    limits = request_limits("mutation", 8192)
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                  timeout=limits["timeout_seconds"], max_retries=limits["sdk_retries"])

def generate_mutation(prompt: str) -> str:
    client = create_mutation_client()
    limits = request_limits("mutation", 8192)
    system_prompt = getSystemPrompt()

    response = client.responses.create(
        model=model_id("MLIRBENCH_MUTATION_MODEL", "gpt-5-mini"),
        max_output_tokens=limits["max_output_tokens"],
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text.strip()