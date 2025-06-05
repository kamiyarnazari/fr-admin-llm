# compare_and_save.py
import os
import csv
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Base model identifier on HF Hub
BASE_MODEL_ID    = "mistralai/Mistral-7B-v0.1"

# Fine-tuned model directory (e.g. "models/mistral-7b-finetuned")
FINE_TUNED_DIR   = r"models/mistral-7b-finetuned"

# Offload folders (one per model) to avoid out-of-memory on local GPU/CPU
OFFLOAD_BASE     = "offload_base"
OFFLOAD_FINE_TUNED = "offload_finetuned"

# Output CSV path
OUTPUT_CSV       = "comparison_results.csv"
# ───────────────────────────────────────────────────────────────────────────────


# 2) List of (question, reference_answer) pairs
TEST_PAIRS = [
    # Standard FAQs:
    (
        "Comment faire une demande de RSA pour une personne isolée ?",
        "Pour faire une demande de RSA en tant que personne isolée, vous devez :\n"
        "1. Rassembler une pièce d’identité valide (carte nationale d’identité ou passeport) et un justificatif de domicile de moins de trois mois (facture EDF, quittance de loyer, etc.).\n"
        "2. Accéder au site de la CAF (www.caf.fr), créer un compte puis remplir le formulaire « Demande de RSA ».\n"
        "3. Joindre vos trois derniers bulletins de salaire ou vos attestations de revenus si vous êtes sans activité.\n"
        "4. Envoyer votre dossier complet en ligne ou le déposer directement dans l’agence CAF de votre département.\n"
        "5. Attendre la réponse de la CAF (environ 1 à 2 semaines) : vous recevrez un courrier ou un e-mail confirmant le montant et la date de versement."
    ),
    (
        "Quels documents sont nécessaires pour renouveler une carte nationale d’identité ?",
        "Pour renouveler une carte nationale d’identité française, il vous faut :\n"
        "1. Votre ancienne carte d’identité (même si elle est périmée depuis moins de cinq ans).\n"
        "2. Une photo d’identité récente et conforme aux normes officielles (fond uni, visage dégagé, expression neutre).\n"
        "3. Un justificatif de domicile de moins de six mois (facture d’électricité, de téléphone, quittance de loyer, etc.).\n"
        "4. Le formulaire CERFA n°12100*02 dûment rempli (disponible en ligne ou en mairie).\n"
        "5. Un timbre fiscal dont le montant varie selon votre âge (25 € pour un adulte, 17 € pour un mineur de moins de 15 ans, etc.).\n"
        "6. Se présenter lors d’un rendez-vous en mairie pour remise des empreintes digitales et validation du dossier."
    ),
    (
        "Comment déclarer mes impôts en ligne sur impots.gouv.fr ?",
        "Pour déclarer vos impôts en ligne :\n"
        "1. Connectez-vous à votre espace personnel sur impots.gouv.fr (créez un compte si nécessaire).\n"
        "2. Dans la rubrique « Déclarer mes revenus », sélectionnez l’année fiscale en cours.\n"
        "3. Vérifiez et complétez les informations préremplies (salaires, pensions, revenus fonciers, etc.).\n"
        "4. Ajoutez vos éventuelles déductions ou crédits d’impôt (dons, frais réels, dépenses pour la transition énergétique, etc.).\n"
        "5. Validez chaque page, signez électroniquement puis envoyez votre déclaration.\n"
        "6. Vous recevrez un accusé de réception immédiatement, et votre avis d’imposition sera disponible sous quelques semaines."
    ),
    (
        "Comment obtenir une carte Vitale ?",
        "Pour obtenir une carte Vitale (Assurance Maladie) :\n"
        "1. Créez un compte personnel sur le site Ameli (www.ameli.fr).\n"
        "2. Dans votre espace, cliquez sur « Commander ma carte Vitale ».\n"
        "3. Remplissez le formulaire en ligne en fournissant votre numéro de sécurité sociale et vos coordonnées.\n"
        "4. Téléchargez ou joignez une copie de votre justificatif d’identité (carte d’identité ou passeport) et un RIB (relevé d’identité bancaire).\n"
        "5. Attendez la réception de la carte Vitale sous 10 à 15 jours : elle vous sera envoyée par la sécurité sociale."
    ),
    (
        "Quels documents faut-il fournir pour une demande d’APL ?",
        "Pour faire une demande d’APL, vous devez fournir :\n"
        "1. Une pièce d’identité (carte d’identité, passeport, titre de séjour).\n"
        "2. Un justificatif de domicile récent (contrat de location ou facture d’électricité).\n"
        "3. Vos trois derniers bulletins de salaire ou, le cas échéant, votre avis de situation Pôle emploi.\n"
        "4. Le formulaire CERFA n°10804*02 rempli (disponible en ligne).\n"
        "5. Un RIB (pour le versement de l’aide).\n"
        "6. Envoyez le dossier complet à la CAF responsable de votre logement : la CAF calculera le montant en fonction de vos ressources et de la situation du logement."
    ),
    # Edge cases:
    (
        "",  # empty question
        "<Pas de question – le modèle devrait demander une reformulation>"
    ),
    (
        "J’ai perdu ma carte Vitale, j’ai déménagé, et j’ai changé de banque. Expliquez toutes les étapes pour déclarer mon changement d’adresse, ré-émettre une carte Vitale, et mettre à jour mes coordonnées bancaires auprès de la CAF.",
        "1. Déclarer changement d’adresse à la CAF (espace en ligne ou courrier).\n"
        "2. Demander le duplicata de carte Vitale (compte Ameli → formulaire « Demander ma carte Vitale »).\n"
        "3. Mettre à jour vos coordonnées bancaires dans votre espace CAF (onglet « Mon compte » → « Gérer mes RIB »)."
    ),
    (
        "Comment obtenir un prêt immobilier en France ?",
        "<Hors sujet – le modèle devrait dire qu’il ne couvre pas ce domaine>"
    ),
    (
        "Commet faires une demande de car Vitale ?",
        "<Typo détectée – le modèle devrait comprendre « carte Vitale » ou demander clarification>"
    ),
    (
        "Comment renouveler mon passeport français ?",
        "Pour renouveler un passeport français :\n"
        "1. Prendre rendez-vous en mairie ou consulat.\n"
        "2. Remplir le formulaire CERFA n°12100*02.\n"
        "3. Fournir l’ancien passeport, photo d’identité, justificatif de domicile, timbre fiscal (86 €).\n"
        "4. Déposer le dossier complet lors du rendez-vous.\n"
        "5. Attendre 2 à 3 semaines pour le nouveau passeport."
    ),
    (
        "Qu’est-ce que je dois faire ?",
        "<Vague – le modèle devrait demander plus de contexte>"
    ),
    (
        "Quels sont les documents nécessaires pour faire une demande de RSA quand on est étudiant boursier, déjà allocataire de la CAF, et que l’on souhaite changer de département ?",
        "1. Pièce d’identité et justificatif de domicile (étudiant boursier : justificatif de bourse).\n"
        "2. Notification de bourse si nécessaire (pour calcul du RSA).\n"
        "3. Preuves de ressources (fiches de paie ou attestation Pôle emploi).\n"
        "4. Déclaration de situation à la CAF (déjà allocataire → mettre à jour le dossier).\n"
        "5. Demander le transfert de dossier CAF vers le nouveau département (mon compte CAF → rubrique « Mes démarches »).\n"
        "6. Envoyer le dossier complet au nouveau relais CAF pour calcul du RSA."
    )
]


# 3) Utility functions

def load_model_and_tokenizer(model_id_or_path: str, offload_folder: str):
    """
    Load a Mistral‐7B‐based model in 4-bit with offloading.
    """
    print(f"\n🔄 Loading tokenizer/model from: {model_id_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(offload_folder, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        offload_folder=offload_folder,
        offload_state_dict=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return tokenizer, model


def generate_answer(question: str, tokenizer, model, max_new_tokens: int = 128) -> str:
    """
    Prepend static French instruction, tokenize, generate, and return only the answer.
    """
    if not question.strip():
        return "<Empty input – model should ask for clarification>"
    prompt = f"Réponds à la question administrative suivante :\n\n{question.strip()}\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


# 4) Main comparison loop + writing CSV

def main():
    # Load both models + tokenizers
    base_tokenizer, base_model = load_model_and_tokenizer(
        BASE_MODEL_ID, offload_folder=OFFLOAD_BASE
    )
    ft_tokenizer, ft_model = load_model_and_tokenizer(
        FINE_TUNED_DIR, offload_folder=OFFLOAD_FINE_TUNED
    )

    # Open CSV for writing
    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow([
            "question",
            "reference_answer",
            "base_model_answer",
            "fine_tuned_model_answer"
        ])

        # Iterate over all test pairs
        for question, reference in TEST_PAIRS:
            # Generate base answer
            try:
                base_ans = generate_answer(question, base_tokenizer, base_model)
            except Exception as e:
                base_ans = f"<Error: {e}>"

            # Generate fine‐tuned answer
            try:
                ft_ans = generate_answer(question, ft_tokenizer, ft_model)
            except Exception as e:
                ft_ans = f"<Error: {e}>"

            # Write one CSV row
            writer.writerow([question, reference, base_ans, ft_ans])
            print(f"✔ Processed: {question[:30]}{'...' if len(question)>30 else ''}")

    print(f"\n Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
