import requests
import csv
import os
from openai import OpenAI
from groq import Groq

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
) 

API_BASE = "https://www.inphoproject.org/idea"
PAIRS_CSV = "pairs.csv"

# with open('pairs.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["idA", "idB"])
# with open('ai_researcher.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["idA", "idB", "relatedness", "generality"])

lst = ["none"]
def idToLabel():
    for id in range(1, 6463):
        print(id)
        try:
            url = f"{API_BASE}/{id}.json"
            r = requests.get(url)
            r.raise_for_status()
            lst.append(r.json()["label"])
        except:
            lst.append("none")
            continue

# idToLabel()

def make_pairs():
    with open('pairs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for id in range(1, 6463):
            try:
                url = f"{API_BASE}/{id}.json"
                r = requests.get(url)
                r.raise_for_status()
            except KeyboardInterrupt:
                raise
            except:
                continue
            
            print(id)
            list1 = r.json()["related"]

            for id2 in list1:
                writer.writerow([lst[id], lst[id2]])

# make_pairs()

### TODO:
# wipe memory of AI after each response
# order seems to not matter when it should for prompting
# add column of model used in csv file
# try at least one more model
# ONCE SEP is collected, then compare on a pair-by basis

def generate(start_line):
    with open("llama_researcher.csv", "a", newline="") as fout:
        writer = csv.writer(fout)

        with open("pairs.csv", newline="") as fin:
            reader = csv.reader(fin)
            next(reader)
            for idx, (idA, idB) in enumerate(reader, start=2):
                if idx < start_line:
                    continue
                try:
                    response = client.chat.completions.create(
                        model = "llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": """
                            You’re a philosophy researcher familiar with the Stanford Encyclopedia of Philosophy.
                    For each pair of ideas, please answer:
                    1) “How related is <Idea A> to <Idea B>?”
                    — Not Related / Marginally Related / Somewhat Related / Related / Highly Related
                    2) If your answer is not “Not Related,” say whether A is More Specific Than / More General Than / As General As / Incomparable To.
                    For both parts, say the answer only concisely.
                            """},
                            {"role": "user", "content": f"How related is {idA} to {idB}?"}
                        ],
                        temperature = 0
                    )
                except KeyboardInterrupt:
                    raise
                except:
                    continue

                text = response.choices[0].message.content.strip()
                related, general = parse_response(text)
                writer.writerow([idA, idB, related, general])

                print(f"{idA} → {idB}: {related}, {general}")

def parse_response(text):
    """
    Returns a tuple (relatedness, generality)
    """
    rel_options = ["Not Related", "Marginally Related", "Somewhat Related", "Related", "Highly Related"]
    gen_options = ["More Specific Than", "More General Than", "As General As", "Incomparable To"]
    
    relatedness = ""
    generality = "N/A"
    lower = text.lower()

    for opt in rel_options:
        if opt.lower() in lower:
            relatedness = opt
            break
    if relatedness != "Not Related":
        for opt in gen_options:
            if opt.lower() in lower:
                generality = opt
                break
    return relatedness, generality


# if "__name__" == "__main__":
#     generate(2)