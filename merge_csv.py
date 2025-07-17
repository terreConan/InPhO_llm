import pandas as pd
import re

REL_MAP = {
    "Not Related": 0, "Marginally Related": 1, "Somewhat Related": 2,
    "Related": 3, "Highly Related": 4,
}
GEN_MAP = {
    "N/A": -1, "Incomparable To": 0, "More Specific Than": 1,
    "As General As": 2,   "More General Than": 3,
}

AI_FILE   = "llama_researcher.csv"
HUM_FILE  = "idea_evaluation_with_all_user_info.csv"
MAP_FILE  = "idea_id_label_mapping.csv"
OUT_FILE  = "expert_ai_pairs.csv"

"""
- loop through rows in HUM_FILE, and regard only the rows with proficiency 4.0
- for each row with this proficiency, look at the ante_id and cons_id (there is also the human relatedness and generality values right next to these columns)
- match these id's to the actual idea in MAP_FILE (ID,label)
- after we get the ideas of ante_id and cons_id, see which row in AI_FILE matches that idA and idB and get the relatedness/generality (idA,idB,relatedness,generality)
- write to OUT_FILE the ideaA, ideaB, aiRelatedness
"""

canon = lambda s: re.sub(r"[^a-z0-9]+","",s.lower()) if isinstance(s,str) else s

m = pd.read_csv(MAP_FILE)
m["canon"] = m["label"].apply(canon)
id2label   = m.set_index("ID")["label"].to_dict()

hum = pd.read_csv(HUM_FILE)
hum = hum[pd.to_numeric(hum["first_area_level"],errors="coerce")==4]

hum["ideaA"] = hum["ante_id"].map(id2label)
hum["ideaB"] = hum["cons_id"].map(id2label)
hum = hum.dropna(subset=["ideaA","ideaB"])
hum["key"]   = hum.apply(lambda r: tuple(sorted((canon(r["ideaA"]),canon(r["ideaB"])))),axis=1)

hum["humanRelatedness"] = pd.to_numeric(hum["relatedness"],errors="coerce")
mask = hum["humanRelatedness"].isna()
hum.loc[mask,"humanRelatedness"] = hum.loc[mask,"relatedness"].map(REL_MAP)

hum["humanGenerality"] = pd.to_numeric(hum["generality"],errors="coerce")
mask = hum["humanGenerality"].isna()
hum.loc[mask,"humanGenerality"] = hum.loc[mask,"generality"].map(GEN_MAP)

hum = hum[["key","ideaA","ideaB","humanRelatedness","humanGenerality"]]

ai = pd.read_csv(AI_FILE).rename(columns={"idA":"ideaA","idB":"ideaB"})
ai["key"] = ai.apply(lambda r: tuple(sorted((canon(r["ideaA"]),canon(r["ideaB"])))),axis=1)
ai["aiRelatedness"] = ai["relatedness"].map(REL_MAP).astype("Int8")
ai["aiGenerality"]  = ai["generality"].map(GEN_MAP).astype("Int8")
ai = ai[["key","aiRelatedness","aiGenerality"]]

out = hum.merge(ai,on="key",how="left")
out = out[["ideaA","ideaB","aiRelatedness","aiGenerality","humanRelatedness","humanGenerality"]]

from relateAI import parse_response
from groq import Groq

client = Groq(
    api_key=""
)

def fill_unknown(idA, idB):
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
    return parse_response(response.choices[0].message.content.strip())

for idx, row in out.iterrows():
    if pd.isna(row["aiRelatedness"]):
        rel_txt, gen_txt = fill_unknown(row["ideaA"], row["ideaB"])

        rel_num = REL_MAP[rel_txt]
        gen_num = GEN_MAP[gen_txt]

        out.at[idx, "aiRelatedness"] = rel_num
        out.at[idx, "aiGenerality"]  = gen_num
        print(f"Filled {row['ideaA']} → {row['ideaB']}: {rel_num}, {gen_num}")

out.to_csv(OUT_FILE,index=False)
print("rows written:",len(out))
