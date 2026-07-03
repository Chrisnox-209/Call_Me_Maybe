from llm_sdk import Small_LLM_Model  # type: ignore
from parse import ParsingPompt, ParsngFunctions
from typing import Any, Iterator, Literal
# import string
# import json


def build_prompt_func(data_function: list[ParsngFunctions]) -> str:
    prompt_func: str = ''
    for func in data_function:
        prompt_func = prompt_func + f"name: {func.name}\n"
        prompt_func = prompt_func + f"description: {func.description}\n"
        prompt_func = prompt_func + "parameters:\n"
        for name, param in func.parameters.items():
            prompt_func = prompt_func + f"  - {name}: {param.type}\n"
        prompt_func = prompt_func + f"returns: {func.returns.type}\n\n"
    return prompt_func


def build_struct() -> str:
    return 'result = [{"prompt": ...,"name": ...,"parameters": {...}}, ...'


def run_inference(data_prompt: list[ParsingPompt],
                  data_function: list[ParsngFunctions],
                  output: str) -> None:
    llm: Any = Small_LLM_Model()

    # prompts: Iterator[ParsingPompt] = iter(data_prompt)
    prompt_func: str = build_prompt_func(data_function)

#    for i in range(10):
#    while True:

#        if j != 0:
    # Decode

    # text: ParsingPompt = next(prompts)
    # prompt: str = text.prompt
    pre_prompt = ('You are an assistant. Here is a question. Here are the '
                  'functions. Complete the rest with the correct information '
                  'to generate a JSON format.\n\n[{"prompt":')

    # struct: str = build_struct()
    llm_prompt: str = (
        "What is the sum of 2 and 3?\n\n" + prompt_func + pre_prompt)
    exit_llm: list[int] = []

    # ENCODAGE
    input_ids: Any = llm.encode(llm_prompt)[0].tolist()

    for etape in range(20):

        # A. On demande les probabilités pour le prochain numéro
        logits: list[float] = llm.get_logits_from_input_ids(input_ids)

        # B. On choisit la probabilité la plus haute (le meilleur token)
        meilleure_proba: float = max(logits)
        prochain_token_id: int = logits.index(meilleure_proba)
        if prochain_token_id == 151643:
            print("Signal de fin")
            break

        # C. On ajoute ce nouveau numéro à notre liste pour le tour suivant
        exit_llm.append(prochain_token_id)
        input_ids.append(prochain_token_id)

        # D. On décode juste ce token pour l'afficher à l'écran
        mot: Any = llm.decode([prochain_token_id])
        # print(f"Meilleur proba: {meilleure_proba}")
        # print(f"   ↳ Token {prochain_token_id} -> '{mot}'")
    # 3. DÉCODAGE FINAL : On retransforme tous les numéros en texte lisible

    reponse_complete: Any = llm.decode(exit_llm)
    print(f"\n✅ Résultat final : {reponse_complete}")
