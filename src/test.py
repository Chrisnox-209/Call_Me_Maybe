from llm_sdk import Small_LLM_Model

def main():
    print("⏳ Chargement du modèle...")
    model = Small_LLM_Model()

    prompt = "The capital of France is"
    print(f"\n🗣️ Prompt de départ : '{prompt}'")
    
    # 1. ENCODAGE : On traduit le texte en liste de numéros (Input IDs)
    input_ids = model.encode(prompt)[0].tolist()
    print(f"🔢 Traduction pour l'IA (Input IDs) : {input_ids}")
    
    print("\n🤖 Génération en cours (5 mots max)...")
    
    # 2. BOUCLE D'INFÉRENCE
    for etape in range(5):
        # A. On demande les probabilités pour le prochain numéro
        logits = model.get_logits_from_input_ids(input_ids)
        
        # B. On choisit la probabilité la plus haute (le meilleur token)
        meilleure_proba = max(logits)
        prochain_token_id = logits.index(meilleure_proba)
        
        # C. On ajoute ce nouveau numéro à notre liste pour le tour suivant
        input_ids.append(prochain_token_id)
        
        # D. On décode juste ce token pour l'afficher à l'écran
        mot = model.decode([prochain_token_id])
        print(f"   ↳ Token {prochain_token_id} -> '{mot}'")

    # 3. DÉCODAGE FINAL : On retransforme tous les numéros en texte lisible
    reponse_complete = model.decode(input_ids)
    print(f"\n✅ Résultat final : {reponse_complete}")

if __name__ == "__main__":
    main()