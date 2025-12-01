"""
Pipeline for generating AI responses using comparison_pairs_with_coverage.csv

Output: A csv file in the processed folder with the columns key, ideaA, 
ideaB, persona, system_prompt, user_prompt, and metadata (JSON string
with model, temperature, date, prompt version, persona.)
"""

import os
import csv
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import anthropic
from mistralai import Mistral
from google import genai
from google.genai import types
from groq import Groq
import time
import random

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
model = "claude-3-7-sonnet-20250219"

client = anthropic.Anthropic(api_key=api_key)

coverage_to_persona = {
    "has_amateur": "novice",
    "has_course_taker": "course_taker",
    "has_expert": "expert",
    "has_phd_student": "phd",
}

rel_options = {
    "Not Related":        0,
    "Marginally Related": 1,
    "Somewhat Related":   2,
    "Related":            3,
    "Highly Related":     4
}

gen_options = {
    "N/A":                -1,
    "More Specific Than": 0,
    "More General Than":  1,
    "As General As":      2,
    "Incomparable To":    3
}

PROMPT_VERSION = "v1"
RANDOM_PERSONA = "random"


def truthy(val):
    """Convert string representations of boolean values to actual booleans."""
    return str(val).strip().lower() in ["1", "true", "t", "yes", "y"]


def load_user_prompt(ideaA, ideaB):
    with open(f"./prompts/{PROMPT_VERSION}/user_prompt.txt", "r") as f:
        base = f.read()
        base = base.replace("{idea_a}", ideaA).replace("{idea_b}", ideaB)
    return base


def load_system_prompt(persona):
    with open(f"./prompts/{PROMPT_VERSION}/system_prompt_{persona}.txt", "r") as f:
        content = f.read()
    return content


def response_to_index(txt):
    txt = txt.lower()
    relate = -1
    general = -1

    if "highly related" in txt:
        relate = 4
    else:
        for key, val in rel_options.items():
            if key.lower() in txt:
                relate = val
                break

    for key, val in gen_options.items():
        if key.lower() in txt:
            general = val
            break

    return relate, general


def call_model(system_prompt: str, user_prompt: str, model: str, client: anthropic.Anthropic):
    """Call Anthropic Claude and return (relatedness_idx, generality_idx)."""
    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    text = ""
    if response.content and len(response.content) > 0:
        text = response.content[0].text
    rel, gen = response_to_index((text or "").strip())
    return rel, gen


def call_mistral(system_prompt: str, user_prompt: str, model: str, client: Mistral):
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )
    time.sleep(1)

    rel, gen = response_to_index(chat_response.choices[0].message.content)
    return rel, gen


def call_gemini(system_prompt: str, user_prompt: str, model: str, client: genai):
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0
        ),
        contents=user_prompt,
    )
    rel, gen = response_to_index(response.text)
    return rel, gen


def call_llama(system_prompt: str, user_prompt: str, model: str, client: Groq):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    rel, gen = response_to_index(completion.choices[0].message.content)
    return rel, gen


def write_response_row(writer, *, key, ideaA, ideaB, persona, rel, gen, model):
    metadata = {
        "model": model,
        "temperature": 0,
        "date": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "persona": persona
    }
    writer.writerow({
        "key": key,
        "ideaA": ideaA,
        "ideaB": ideaB,
        "relatedness": rel,
        "generality": gen,
        "persona": persona,
        "system_prompt": PROMPT_VERSION,
        "user_prompt": PROMPT_VERSION,
        "metadata": json.dumps(metadata, ensure_ascii=False),
    })


def main():
    """
    Pipeline; generates CSV in output_f.
    """
    random.seed(42)

    output_f = "./data/processed/v3/claude_outputs_v3.csv"
    input_f = "./data/processed/comparison_pairs_with_coverage.csv"

    fieldnames = [
        "key", "ideaA", "ideaB",
        "relatedness", "generality",
        "persona", "system_prompt", "user_prompt", "metadata"
    ]

    with open(output_f, "w") as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()

        with open(input_f, "r") as input_csv:
            reader = csv.DictReader(input_csv)

            for pair in reader:
                key = pair.get("key")
                ideaA = pair.get("ideaA")
                ideaB = pair.get("ideaB")

                user_prompt = load_user_prompt(ideaA, ideaB)

                # Persona-based LLM runs (novice, course_taker, expert, phd)
                for coverage, persona in coverage_to_persona.items():
                    if truthy(pair.get(coverage)):
                        system_prompt = load_system_prompt(persona)
                        rel, gen = call_model(system_prompt, user_prompt, model, client)
                        write_response_row(
                            writer,
                            key=key, ideaA=ideaA, ideaB=ideaB,
                            persona=persona, rel=rel, gen=gen, model=model
                        )

                # Random baseline: one random output per row
                rel = random.choice(list(rel_options.values()))
                gen = random.choice(list(gen_options.values()))
                write_response_row(
                    writer,
                    key=key, ideaA=ideaA, ideaB=ideaB,
                    persona=RANDOM_PERSONA, rel=rel, gen=gen, model="random"
                )

    print("Output finished in " + output_f)


if __name__ == "__main__":
    main()

