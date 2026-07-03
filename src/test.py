from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Any
import numpy as np
from numpy.typing import NDArray
# import json


def main() -> None:
    llm: Any = Small_LLM_Model(device="cpu")

    questions: list[str] = [
        "Est-ce que le soleil est une étoile ?",
        "Est-ce que l’eau est liquide à température ambiante ?",
        "Est-ce que Paris est en France ?",
        "Est-ce que 2 est un nombre pair ?",
        "Est-ce que les humains respirent de l’oxygène ?",
        "Est-ce que la Terre est une planète ?",
        "Est-ce que la Lune tourne autour de la Terre ?",
        "Est-ce que le feu produit de la chaleur ?",
        "Est-ce que 10 est plus grand que 5 ?",
        "Est-ce que les chats sont des animaux ?"
    ]

    # path_vocab: str = llm.get_path_to_vocab_file()
    # with open(path_vocab, "r", encoding="utf-8") as file:
    #     vocabulaire: Any = json.load(file)

    tokens_autorises_ids: list[Any] = [
        llm.encode("oui")[0][0],
        llm.encode("non")[0][0]
    ]

    for i in range(len(questions)):

        # Prompt système simple
        # pre_prompt = (
        #     "Tu es un modèle d’IA et tu dois répondre "
        #     "uniquement par oui ou par non."
        # )
        pre_prompt = (
            "Réponds uniquement selon la vérité factuelle.\n"
            "Réfléchis avant de répondre."
        )

        prompt: str = pre_prompt + "\n\n" + questions[i]

        # 6. Encodage du prompt
        input_ids: Any = llm.encode(prompt)[0].tolist()

        # 7. Récupération des logits
        logits: NDArray[Any] = np.array(llm.get_logits_from_input_ids(
            input_ids))

        # 8. MASKING DES LOGITS
        # On met tous les tokens à -inf sauf oui/non
        masked_logits: NDArray[Any] = np.full_like(logits, -np.inf)

        # On garde uniquement les logits des tokens autorisés
        masked_logits[tokens_autorises_ids] = logits[tokens_autorises_ids]
        for t in tokens_autorises_ids:
            print(t, logits[t])

        # 9. Sélection du meilleur token
        prochain_token_id = int(np.argmax(masked_logits))

        # 10. Décodage du token
        result: Any = llm.decode(prochain_token_id)

        if "oui" in result.lower():
            color = "\033[32m"
        elif "non" in result.lower():
            color = "\033[31m"
        else:
            color = "\033[0m"
        print(f"{color}[{result}]\033[0m --> {questions[i]}")


if __name__ == "__main__":
    main()
