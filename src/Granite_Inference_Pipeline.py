#!/usr/bin/env python3
"""Benchmark d'inférence pour les modèles IBM Granite (transformers + PyTorch).

Mesure, avec synchronisation GPU correcte :
  - TTFT (time to first token = prefill + 1er token)
  - latence par token en phase de décodage
  - débit de décodage et débit de bout en bout
  - VRAM résidente (modèle) et pic de VRAM pendant la génération

Les résultats sont exportés en JSON + CSV (+ graphique PNG), et un profil
torch.profiler peut être généré en option.

Exemples :
    python src/Granite_Inference_Pipeline.py
    python src/Granite_Inference_Pipeline.py --dtype bfloat16 --repeats 10 --batch-size 4
    python src/Granite_Inference_Pipeline.py --profile --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "ibm-granite/granite-3.0-2b-instruct"
DEFAULT_PROMPT = (
    "Explique le fonctionnement d'un régulateur PID (Proportionnel, Intégral, Dérivé) "
    "pour le contrôle d'un système automatique."
)
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

EXIT_OK = 0
EXIT_OOM = 2
EXIT_LOAD_ERROR = 3


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Identifiant Hugging Face ou chemin local")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", *DTYPES],
        help="auto = bfloat16 si supporté, sinon float16 sur GPU ; float32 sur CPU",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", type=Path, help="Fichier texte contenant le prompt (prioritaire sur --prompt)")
    p.add_argument("--no-chat-template", action="store_true", help="Passe le prompt brut, sans chat template")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2, help="Générations de chauffe (même config que la mesure)")
    p.add_argument("--repeats", type=int, default=5, help="Nombre de mesures")
    p.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Autorise l'arrêt sur EOS (sinon la longueur générée est fixée pour comparer les runs)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--no-plots", action="store_true", help="Désactive le graphique matplotlib")
    p.add_argument("--profile", action="store_true", help="Exporte une trace torch.profiler")
    p.add_argument("--profile-tokens", type=int, default=32, help="Tokens générés pendant le profilage")
    args = p.parse_args(argv)

    for name in ("max_new_tokens", "batch_size", "repeats"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} doit être >= 1")
    if args.warmup < 0:
        p.error("--warmup doit être >= 0")
    return args


def resolve_device_dtype(args: argparse.Namespace) -> tuple[str, torch.dtype]:
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA demandé mais non disponible (torch.cuda.is_available() == False).")
    if device == "cpu" and args.device == "auto":
        print("ATTENTION : CUDA n'est pas détecté. Le benchmark tourne sur CPU (très lent).")

    if args.dtype == "auto":
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = DTYPES[args.dtype]

    if device == "cpu" and dtype == torch.float16:
        print("ATTENTION : float16 est mal supporté sur CPU, bascule en float32.")
        dtype = torch.float32
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("ATTENTION : bfloat16 non supporté par ce GPU, bascule en float16.")
        dtype = torch.float16
    return device, dtype


# --------------------------------------------------------------------------- #
# Utilitaires GPU / mémoire
# --------------------------------------------------------------------------- #
def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def mb(n_bytes: float) -> float:
    return n_bytes / (1024**2)


def cleanup(*objects) -> None:
    """Libère les objets passés puis vide les caches (appelé dans `finally`)."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Chargement et préparation des entrées
# --------------------------------------------------------------------------- #
def load_model(model_id: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # indispensable pour la génération par batch

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def build_inputs(tokenizer, prompt: str, batch_size: int, use_chat_template: bool, device: str):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        add_special = False  # le template insère déjà les tokens spéciaux
    else:
        text = prompt
        add_special = True
    inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, add_special_tokens=add_special)
    return {k: v.to(device) for k, v in inputs.items()}


# --------------------------------------------------------------------------- #
# Mesure
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    run: int
    prompt_tokens: int
    new_tokens: int  # par séquence
    batch_size: int
    ttft_ms: float
    total_ms: float
    decode_ms_per_token: float
    decode_tokens_per_s: float  # tous éléments du batch confondus
    e2e_tokens_per_s: float
    peak_vram_mb: float


def timed_generate(model, inputs: dict, gen_kwargs: dict, device: str):
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    sync(device)
    return out, time.perf_counter() - t0


def make_gen_kwargs(tokenizer, n_tokens: int, fixed_length: bool) -> dict:
    kwargs = dict(max_new_tokens=n_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    if fixed_length:
        kwargs["min_new_tokens"] = n_tokens
    return kwargs


def measure_run(model, tokenizer, inputs: dict, args, device: str, run_idx: int) -> RunResult:
    batch = inputs["input_ids"].shape[0]
    prompt_len = inputs["input_ids"].shape[1]
    fixed = not args.allow_early_stop

    # 1) TTFT : prefill + 1 token
    _, ttft_s = timed_generate(model, inputs, make_gen_kwargs(tokenizer, 1, fixed), device)

    # 2) Génération complète, avec pic de VRAM mesuré uniquement sur cette phase
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out, total_s = timed_generate(model, inputs, make_gen_kwargs(tokenizer, args.max_new_tokens, fixed), device)
    peak = mb(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

    new_tokens = out.shape[1] - prompt_len
    decode_steps = max(new_tokens - 1, 0)
    decode_s = max(total_s - ttft_s, 1e-9)
    return RunResult(
        run=run_idx,
        prompt_tokens=prompt_len,
        new_tokens=new_tokens,
        batch_size=batch,
        ttft_ms=ttft_s * 1000,
        total_ms=total_s * 1000,
        decode_ms_per_token=(decode_s / decode_steps) * 1000 if decode_steps else float("nan"),
        decode_tokens_per_s=(batch * decode_steps) / decode_s if decode_steps else float("nan"),
        e2e_tokens_per_s=(batch * new_tokens) / total_s,
        peak_vram_mb=peak,
    )


def summarize(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_runs(runs: list[RunResult]) -> dict:
    metrics = ["ttft_ms", "total_ms", "decode_ms_per_token", "decode_tokens_per_s", "e2e_tokens_per_s", "peak_vram_mb"]
    return {m: summarize([getattr(r, m) for r in runs]) for m in metrics}


# --------------------------------------------------------------------------- #
# Profilage et export
# --------------------------------------------------------------------------- #
def run_profiler(model, tokenizer, inputs: dict, args, device: str, out_dir: Path) -> None:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    kwargs = make_gen_kwargs(tokenizer, args.profile_tokens, fixed_length=not args.allow_early_stop)
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with torch.inference_mode():
            model.generate(**inputs, **kwargs)
        sync(device)

    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print("\n--- Top 15 opérateurs (torch.profiler) ---")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    print(f"Trace Chrome exportée : {trace_path} (à ouvrir dans chrome://tracing ou ui.perfetto.dev)")


def save_results(out_dir: Path, meta: dict, runs: list[RunResult], summary: dict) -> None:
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "runs": [asdict(r) for r in runs]}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(runs[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in runs)


def plot_results(out_dir: Path, runs: list[RunResult], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib indisponible : graphique ignoré.")
        return

    xs = [r.run for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(xs, [r.ttft_ms for r in runs], color="tab:blue")
    axes[0].set(title="TTFT", xlabel="run", ylabel="ms")
    axes[1].bar(xs, [r.decode_ms_per_token for r in runs], color="tab:orange")
    axes[1].set(title="Latence par token (décodage)", xlabel="run", ylabel="ms/token")
    axes[2].bar(xs, [r.decode_tokens_per_s for r in runs], color="tab:green")
    axes[2].set(title="Débit de décodage", xlabel="run", ylabel="tokens/s")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark.png", dpi=150)
    plt.close(fig)


def collect_meta(args, device: str, dtype: torch.dtype, vram_model_mb: float, prompt_tokens: int) -> dict:
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "fixed_length": not args.allow_early_stop,
        "chat_template": not args.no_chat_template,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vram_model_mb": vram_model_mb,
    }
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name()
        meta["cuda"] = torch.version.cuda
    return meta


def print_report(meta: dict, summary: dict) -> None:
    print("\n=== RÉSULTATS DU BENCHMARK ===")
    print(f"Modèle      : {meta['model']} ({meta['dtype']}) sur {meta.get('gpu', meta['device'])}")
    print(f"Config      : batch={meta['batch_size']} | prompt={meta['prompt_tokens']} tok | "
          f"génération={meta['max_new_tokens']} tok | {meta['repeats']} mesures")
    if meta["device"] == "cuda":
        print(f"VRAM modèle : {meta['vram_model_mb']:.0f} MB")
    rows = [
        ("TTFT (ms)", "ttft_ms"),
        ("Latence totale (ms)", "total_ms"),
        ("Latence/token décodage (ms)", "decode_ms_per_token"),
        ("Débit décodage (tok/s)", "decode_tokens_per_s"),
        ("Débit bout en bout (tok/s)", "e2e_tokens_per_s"),
        ("Pic VRAM génération (MB)", "peak_vram_mb"),
    ]
    print(f"\n{'Métrique':<30}{'médiane':>11}{'moyenne':>11}{'écart-type':>12}{'min':>11}{'max':>11}")
    for label, key in rows:
        s = summary[key]
        print(f"{label:<30}{s['median']:>11.2f}{s['mean']:>11.2f}{s['std']:>12.2f}{s['min']:>11.2f}{s['max']:>11.2f}")


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
def run_benchmark(args: argparse.Namespace) -> int:
    device, dtype = resolve_device_dtype(args)
    torch.manual_seed(args.seed)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt

    tokenizer = model = inputs = None
    try:
        print(f"Chargement de {args.model} ({str(dtype).replace('torch.', '')}) sur {device}...")
        if device == "cuda":
            torch.cuda.empty_cache()
            vram_before = torch.cuda.memory_allocated()
        try:
            tokenizer, model = load_model(args.model, device, dtype)
        except torch.cuda.OutOfMemoryError:
            raise
        except (OSError, ValueError) as exc:
            print(f"ERREUR de chargement du modèle '{args.model}' : {exc}", file=sys.stderr)
            return EXIT_LOAD_ERROR

        vram_model = mb(torch.cuda.memory_allocated() - vram_before) if device == "cuda" else 0.0
        inputs = build_inputs(tokenizer, prompt, args.batch_size, not args.no_chat_template, device)
        prompt_tokens = inputs["input_ids"].shape[1]
        fixed = not args.allow_early_stop

        print(f"Chauffe ({args.warmup} génération(s))...")
        for _ in range(args.warmup):
            timed_generate(model, inputs, make_gen_kwargs(tokenizer, args.max_new_tokens, fixed), device)

        runs: list[RunResult] = []
        for i in range(1, args.repeats + 1):
            result = measure_run(model, tokenizer, inputs, args, device, i)
            runs.append(result)
            print(f"  run {i}/{args.repeats} : TTFT {result.ttft_ms:.1f} ms | "
                  f"{result.decode_tokens_per_s:.1f} tok/s (décodage)")

        meta = collect_meta(args, device, dtype, vram_model, prompt_tokens)
        summary = summarize_runs(runs)
        print_report(meta, summary)

        model_slug = args.model.replace("/", "_")
        out_dir = args.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{model_slug}_{meta['dtype']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_results(out_dir, meta, runs, summary)
        if not args.no_plots:
            plot_results(out_dir, runs, f"{args.model} | {meta['dtype']} | batch {args.batch_size}")
        if args.profile:
            run_profiler(model, tokenizer, inputs, args, device, out_dir)
        print(f"\nRésultats enregistrés dans : {out_dir}")
        return EXIT_OK

    except torch.cuda.OutOfMemoryError:
        print("ERREUR : mémoire GPU insuffisante (OOM). Réduis --batch-size / --max-new-tokens, "
              "ou utilise --dtype float16/bfloat16.", file=sys.stderr)
        return EXIT_OOM
    finally:
        cleanup(model, tokenizer, inputs)


def main() -> None:
    sys.exit(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
