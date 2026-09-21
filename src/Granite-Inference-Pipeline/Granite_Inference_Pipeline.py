import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_benchmark(model_id="ibm-granite/granite-3.0-2b-instruct", device="cuda"):
    print(f"Chargement du modèle {model_id} sur {device}...")
    
    # Suivi de la VRAM avant chargement
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024**2)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=device, 
        torch_dtype=torch.float16 # Chargement standard en 16-bit
    )

    if device == "cuda":
        vram_after_load = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"VRAM utilisée par le modèle : {vram_after_load - vram_before:.2f} MB")

    # Prompt de test 
    prompt = "Explique le fonctionnement d'un régulateur PID (Proportionnel, Intégral, Dérivé) pour le contrôle d'un système automatique."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\nGénération en cours...")
    
    # Phase d'échauffement (Warm-up) pour éviter de fausser le chrono avec l'initialisation de CUDA
    _ = model.generate(**inputs, max_new_tokens=10)

    # Chronométrage de l'inférence réelle
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False, # Température à 0 pour avoir toujours le même texte
            pad_token_id=tokenizer.eos_token_id
        )
        
    end_time = time.time()
    
    # Calcul des métriques
    total_time = end_time - start_time
    # On ne compte que les nouveaux tokens générés, pas le prompt
    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    throughput = generated_tokens / total_time

    print(f"\n--- RÉSULTATS DU BENCHMARK ---")
    print(f"Temps de réponse (Latence globale) : {total_time:.2f} secondes")
    print(f"Tokens générés : {generated_tokens}")
    print(f"Débit (Throughput) : {throughput:.2f} tokens/seconde")
    
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Pic de VRAM total pendant l'inférence : {peak_vram:.2f} MB")

if __name__ == "__main__":
    # Vérification de la disponibilité du GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ATTENTION : CUDA n'est pas détecté. Le benchmark va tourner sur CPU (très lent).")
    
    run_benchmark(device=device)