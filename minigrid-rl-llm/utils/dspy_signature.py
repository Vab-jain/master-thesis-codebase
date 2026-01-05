import dspy
from typing import Literal
import json
import utils

def configure_llm(llm_model_id='llama-3.3-70b-versatile', cache_llm_dspy=False, GROQ=True):
    if GROQ:
        lm = dspy.LM(llm_model_id, api_base='https://api.groq.com/openai/v1', api_key=utils.GROQ_API_KEY, cache=cache_llm_dspy)
    else:
        lm = dspy.LM(llm_model_id, api_base='http://localhost:11434', cache=cache_llm_dspy)
    dspy.configure(lm=lm)

class NextStepSignature(dspy.Signature):
    """
    Predicts either primitive action or subgoal based on mode.
    """
    task_description: str = dspy.InputField(desc="Task description")
    current_state: str = dspy.InputField(desc="Textual encoding of the env state")
    previous_actions: str = dspy.InputField(desc="List of previous actions taken by the agent in format 'step-1: 2, step-2: 0, step-3: 2'", default="")

    # Output fields based on mode
    primitive_action: int = dspy.OutputField(desc="An integer between 0 and 6 where: 0 Turn left; 1 Turn right; 2 Move forward; 3 Pick up; 4 Drop; 5 Toggle; 6 Done", default=None)

class SubgoalPredictor(dspy.Module):
    """Predictor that outputs either primitive action or subgoal."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(NextStepSignature)

    def forward(self, task_description, current_state, previous_actions):
        pred = self.predictor(task_description=task_description, current_state=current_state, previous_actions=previous_actions)
        return pred

# ─── Stand-alone Test Harness ───────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_FILE = "GT_dataset_BabyAIBot/GT_text_encodings/combined_train.json"
    with open(DATA_FILE, "r") as f:
        dataset = json.load(f)

    configure_llm()
    predictor = SubgoalPredictor()

    for i, entry in enumerate(dataset):
        task_desc    = entry["task_description"]
        # choose whichever encoding you like: "natural", "ascii", "tuples", or "relative"
        current_state = entry["encodings"]["natural"]
        previous_actions = ""  # Initialize with empty string for testing (no previous actions)

        resp = predictor.forward(
            task_description=task_desc,
            current_state=current_state,
            previous_actions=previous_actions
        )

        print(f"\n––– Sample {i+1} –––")
        print(f"Primitive Action : {resp.primitive_action}")
        
        # stop after 5 samples for a smoke test
        if i >= 4:
            break
