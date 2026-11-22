DATA_DIR = "data/erisk25-t1-dataset"
PROCESSED_DATA_DIR = "data/erisk25-t1-processed"
DESCRIPTION_FOLDER = "clustering/questionaire"

MODEL_BASELINE = "rafalposwiata/deproberta-large-depression"
CHECKPOINT_PATH = "checkpoints/checkpoint-v0"
RELEVANT_THRESHOLD = 0.5

LLM_DESCRIPTION_FOLDER = "<your_path_to_LLM_descriptions>"  # <symptom>.txt, five statements per line
NOMIC_TASK_PREFIX = "clustering: "
NOISE_THRESHOLD = 0.60
KMEANS_K = 21

LIST_BDI_SYMPTOMS = [
    "sadness",
    "pessimism",
    "past_failure",
    "loss_of_pleasure",
    "guilty_feelings",
    "punishment",
    "self-dislike",
    "self-criticalness",
    "suicidal",
    "crying",
    "agitation",
    "loss_of_interest",
    "indecisiveness",
    "worthlessness",
    "loss_of_energy",
    "sleeping_pattern",
    "irritability",
    "appetite",
    "concentration",
    "tiredness",
    "loss_of_interest_in_sex",
]

BDI_SYMPTOM_NAMES = {
    "sadness": "1. Sadness",
    "pessimism": "2. Pessimism",
    "past_failure": "3. Past Failure",
    "loss_of_pleasure": "4. Loss of Pleasure",
    "guilty_feelings": "5. Guilty Feelings",
    "punishment": "6. Punishment Feelings",
    "self-dislike": "7. Self-Dislike",
    "self-criticalness": "8. Self-Criticalness",
    "suicidal": "9. Suicidal Thoughts or Wishes",
    "crying": "10. Crying",
    "agitation": "11. Agitation",
    "loss_of_interest": "12. Loss of Interest",
    "indecisiveness": "13. Indecisiveness",
    "worthlessness": "14. Worthlessness",
    "loss_of_energy": "15. Loss of Energy",
    "sleeping_pattern": "16. Changes in Sleeping Pattern",
    "irritability": "17. Irritability",
    "appetite": "18. Changes in Appetite",
    "concentration": "19. Concentration Difficulty",
    "tiredness": "20. Tiredness or Fatigue",
    "loss_of_interest_in_sex": "21. Loss of Interest in Sex",
}

LABELED_SENTENCE_DIR = "data/erisk23-24-sentences"
