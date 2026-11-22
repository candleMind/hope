import os

from utils.constants import BDI_SYMPTOM_NAMES, LLM_DESCRIPTION_FOLDER

PROMPT = """
You are a medical AI assistant specialized in generating realistic symptom descriptions for clinical training data.

SYMPTOM INFORMATION:
- Symptom Name: {symptom_name}
- Definition: {symptom_definition}
- Example Descriptions: {examples}

TASK:
Generate diverse symptom descriptions that capture the full spectrum of this symptom's manifestation.

REQUIREMENTS:
1. Use FIRST-PERSON perspective (start with "I feel", "I experience", "I have", etc.)
2. Generate at least 5 distinct statements
3. Cover varying severity levels (mild/occasional → very severe/continuous)
4. Include different manifestations of the symptom (e.g., if it's sleeping disorder, include both insomnia and hypersomnia)
5. Each statement should be concise but descriptive
6. Focus on realistic, human-experience descriptions

OUTPUT FORMAT:
- One statement per line
- No numbering or bullet points
- Natural, conversational language

EXAMPLES OF GOOD OUTPUT STRUCTURE:
I feel slightly irritated by small things that normally wouldn't bother me.
Some days I find myself sleeping 10+ hours and still feeling exhausted.
I've been waking up at 3 AM every night and can't get back to sleep.
My appetite has completely disappeared and I've lost interest in eating.
The fatigue is so overwhelming that even basic tasks feel impossible.

Now generate for {symptom_name}:
"""


def generate(prompt):
    """Call the LLM with prompt, return list of statements (one per line)."""
    # TODO: implement LLM call with your choice of LLM service
    raise NotImplementedError


def generate_descriptions(
    symptoms, out_dir=LLM_DESCRIPTION_FOLDER, definitions=None, examples=None
):
    """Generate statements for each symptom key and write to out_dir/<symptom>.txt.

    definitions / examples: optional dict symptom -> str.
    """
    if not out_dir or out_dir.startswith("<"):
        raise ValueError("Set LLM_DESCRIPTION_FOLDER in utils/constants.py before generating descriptions.")
    definitions, examples = definitions or {}, examples or {}
    os.makedirs(out_dir, exist_ok=True)
    for symptom in symptoms:
        prompt = PROMPT.format(
            symptom_name=BDI_SYMPTOM_NAMES.get(symptom, symptom),
            symptom_definition=definitions.get(symptom, ""),
            examples=examples.get(symptom, ""),
        )
        lines = [l.strip() for l in generate(prompt) if l.strip()]
        with open(os.path.join(out_dir, f"{symptom}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
