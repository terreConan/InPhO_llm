"""
Pipeline for generating AI responses using comparison_pairs_with_coverage.csv

Output: A csv file in the processed folder with the columns key, ideaA, 
ideaB, persona, system_prompt, user_prompt, and metadata (JSON string
with model, temperature, date, prompt version, persona.)
"""

import os
import csv
import json
from mistralai import Mistral
from google import genai
from google.genai import types
from datetime import datetime, timezone

# Tested using Mistral:
# api_key = os.environ["MISTRAL_API_KEY"]
# model = "open-mistral-nemo"

# client = Mistral(api_key=api_key)

api_key = os.environ["GEMINI_API_KEY"]
model = "gemini-2.5-pro"

client = genai.Client(api_key=api_key)

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
    "N/A":               -1,
    "More Specific Than": 0,
    "More General Than":  1,
    "As General As":      2,
    "Incomparable To":    3
}
PROMPT_VERSION = "v1"

def truthy(val):
    """Convert string representations of boolean values to actual booleans"""
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


def main():
    """
    pipeline; generated CSV in output_f
    """
    output_f = "./data/processed/gemini-pro_outputs_v1.csv"
    input_f = "./data/processed/comparison_pairs_with_coverage.csv"
    
    fieldnames = ["key", "ideaA", "ideaB", "relatedness", "generality", "persona", "system_prompt", "user_prompt", "metadata"]
    with open(output_f, "w") as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()

        with open(input_f, "r") as input_csv:
            reader = csv.DictReader(input_csv)

            for pair in reader:
                key   = pair.get("key")
                ideaA = pair.get("ideaA")
                ideaB = pair.get("ideaB")

                user_prompt = load_user_prompt(ideaA, ideaB)
                

                for coverage, persona in coverage_to_persona.items():
                    if truthy(pair.get(coverage)):
                        system_prompt = load_system_prompt(persona)
                        
                        response = client.models.generate_content(
                            model="gemini-2.5-pro",
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.0),
                            contents=user_prompt,
                        )
                        
                        rel, gen = response_to_index(response.text)

                        metadata = {
                            "model": "gemini-2.5-pro",
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
                        

    print("Output finished in " + output_f)

if __name__ == "__main__":
    main()