#Jinsy
from ai_workflow.model_loader import load_model, list_models

list_models()                      # check what fits
llm = load_model()                 # auto-pick the best model
# llm = load_model("mistral-7b")   # or force a specific one
print(llm.chat([{"role": "user", "content": "Summarise this patent..."}]))