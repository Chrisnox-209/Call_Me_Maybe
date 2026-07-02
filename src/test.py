from llm_sdk import Small_LLM_Model
from typing import Any


def main() -> None:
    print("Chargement du modèle...")
    llm: Any = Small_LLM_Model(device="cpu")

    prompt = "La capitale de la France est"
    print(f"Prompt de départ : '{prompt}'")

    # 1. ENCODAGE : On traduit le texte en liste de numéros (Input IDs)
    input_ids: Any = llm.encode(prompt)[0].tolist()
    print(f"Traduction pour l'IA (Input IDs) : {input_ids}")

    print("Génération en cours (5 mots max)...")
    # 2. BOUCLE D'INFÉRENCE
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
        input_ids.append(prochain_token_id)

        # D. On décode juste ce token pour l'afficher à l'écran
        mot: Any = llm.decode([prochain_token_id])
        print(f"Meilleur proba: {meilleure_proba}")
        print(f"   ↳ Token {prochain_token_id} -> '{mot}'")

    # 3. DÉCODAGE FINAL : On retransforme tous les numéros en texte lisible
    reponse_complete: Any = llm.decode(input_ids)
    print(f"\n✅ Résultat final : {reponse_complete}")


if __name__ == "__main__":
    main()
