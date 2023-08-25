from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

config = PeftConfig.from_pretrained("bogeumkim/polyglot-1.3b-qlora-emotion-classification")

BASE_MODEL = "EleutherAI/polyglot-ko-1.3b"
PEFT_MODEL = "bogeumkim/polyglot-1.3b-qlora-emotion-classification"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
peft_model = PeftModel.from_pretrained(model, PEFT_MODEL)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def gen(x):
    q = f"### 질문: {x}\n### 답변:"

    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False
        ),
        
        max_new_tokens=3,
        temperature=0.001,
        do_sample=True,
        eos_token_id=2,
    )

    result = tokenizer.decode(gened[0]).split('\n')[1]
    result = result.split(':')[1]

    return result.strip()