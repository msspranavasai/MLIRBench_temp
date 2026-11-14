from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from agents.mutation_agent.prompts.system_prompt import getSystemPrompt

client = OpenAI()

def generate_mutation(prompt: str) -> str:
    system_prompt = getSystemPrompt()

    response = client.responses.create(
        model="gpt-5-mini",
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