from llm_sdk.llm_sdk import Small_LLM_Model

def main():
    print("⏳ Chargement du modèle...")
    model = Small_LLM_Model()

    texte_brut = "The capital of France is"
    
    print("\n🚨 Tentative d'envoi du texte brut au moteur...")
    # On essaie de contourner l'encodage !
    logits = model.get_logits_from_input_ids(texte_brut) 
    
    print("Si tu lis ça, c'est que j'avais tort !")

if __name__ == "__main__":
    main()