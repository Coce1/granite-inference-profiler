"""Tests du benchmark d'inférence Granite.

Niveaux de tests (sélectionnables avec `-m`) :
  - (aucun marqueur) : tests unitaires rapides, sans GPU ni téléchargement (modèle factice).
  - integration      : télécharge un mini-modèle Hugging Face et lance le vrai pipeline.
  - gpu              : nécessite un GPU CUDA (mémoire, dtype, métadonnées GPU).
  - slow             : lance le vrai modèle Granite (exige RUN_GRANITE=1 et un GPU).

Exemples :
    pytest tests -v -m "not integration and not gpu and not slow"
    pytest tests -v -m "integration or gpu"
    RUN_GRANITE=1 pytest tests -v -m slow
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import Granite_Inference_Pipeline as gip

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "src" / "Granite_Inference_Pipeline.py"
TINY_MODEL = os.environ.get("TEST_TINY_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
HAS_CUDA = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Doublures (fakes) : un faux tokenizer et un faux modèle, 100 % CPU
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, chat_template="tpl"):
        self.chat_template = chat_template
        self.last_text = None
        self.last_add_special = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return "<chat>" + messages[0]["content"]

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        self.last_text = texts
        self.last_add_special = add_special_tokens
        n = len(texts)
        return {
            "input_ids": torch.ones(n, 5, dtype=torch.long),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
        }


class FakeModel:
    """Imite model.generate : ajoute max_new_tokens tokens et prend un temps proportionnel."""

    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, min_new_tokens=None,
                 do_sample=False, pad_token_id=0):
        self.calls.append({"max_new_tokens": max_new_tokens, "min_new_tokens": min_new_tokens})
        time.sleep(0.005 * max_new_tokens)
        new = torch.ones(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


def make_inputs(batch=2, length=5):
    return {
        "input_ids": torch.ones(batch, length, dtype=torch.long),
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
    }


def make_run(i=1, ttft=10.0, per_tok=5.0):
    return gip.RunResult(
        run=i, prompt_tokens=5, new_tokens=8, batch_size=1,
        ttft_ms=ttft, total_ms=ttft + 7 * per_tok,
        decode_ms_per_token=per_tok, decode_tokens_per_s=1000.0 / per_tok,
        e2e_tokens_per_s=100.0, peak_vram_mb=0.0,
    )


def cpu_args(tmp_path, *extra):
    return gip.parse_args([
        "--device", "cpu", "--output-dir", str(tmp_path),
        "--warmup", "1", "--repeats", "2", "--max-new-tokens", "8", "--no-plots", *extra,
    ])


def read_results(tmp_path):
    files = list(tmp_path.glob("*/results.json"))
    assert len(files) == 1, f"results.json attendu une fois, trouvé : {files}"
    return json.loads(files[0].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# parse_args
# --------------------------------------------------------------------------- #
def test_parse_args_defaults():
    args = gip.parse_args([])
    assert args.max_new_tokens == 256
    assert args.repeats == 5
    assert args.batch_size == 1
    assert args.warmup == 2
    assert args.dtype == "auto"
    assert args.device == "auto"
    assert args.seed == 42
    assert args.allow_early_stop is False
    assert args.profile is False


@pytest.mark.parametrize("argv", [
    ["--repeats", "0"],
    ["--batch-size", "0"],
    ["--max-new-tokens", "0"],
    ["--warmup", "-1"],
    ["--dtype", "int3"],
    ["--device", "tpu"],
])
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        gip.parse_args(argv)


# --------------------------------------------------------------------------- #
# Device / dtype
# --------------------------------------------------------------------------- #
def test_resolve_cpu_defaults_to_float32():
    device, dtype = gip.resolve_device_dtype(SimpleNamespace(device="cpu", dtype="auto"))
    assert (device, dtype) == ("cpu", torch.float32)


def test_resolve_cpu_float16_falls_back_to_float32():
    device, dtype = gip.resolve_device_dtype(SimpleNamespace(device="cpu", dtype="float16"))
    assert (device, dtype) == ("cpu", torch.float32)


def test_resolve_auto_without_cuda_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, dtype = gip.resolve_device_dtype(SimpleNamespace(device="auto", dtype="auto"))
    assert (device, dtype) == ("cpu", torch.float32)


def test_resolve_cuda_requested_but_unavailable_exits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        gip.resolve_device_dtype(SimpleNamespace(device="cuda", dtype="auto"))


@pytest.mark.parametrize("requested", ["auto", "bfloat16"])
def test_resolve_cuda_without_bf16_uses_float16(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gip, "bf16_native", lambda: False)
    device, dtype = gip.resolve_device_dtype(SimpleNamespace(device="cuda", dtype=requested))
    assert (device, dtype) == ("cuda", torch.float16)


def test_resolve_cuda_with_bf16_uses_bfloat16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gip, "bf16_native", lambda: True)
    device, dtype = gip.resolve_device_dtype(SimpleNamespace(device="cuda", dtype="auto"))
    assert (device, dtype) == ("cuda", torch.bfloat16)


def test_bf16_native_false_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gip.bf16_native() is False


@pytest.mark.parametrize("capability,expected", [((7, 5), False), ((8, 0), True), ((8, 9), True), ((9, 0), True)])
def test_bf16_native_depends_on_compute_capability(monkeypatch, capability, expected):
    """T4 = 7.5 -> pas de BF16 matériel ; A100 = 8.0, L4 = 8.9, H100 = 9.0 -> oui."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert gip.bf16_native() is expected


def test_resolve_t4_like_gpu_picks_float16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    # même si torch annonce un BF16 "émulé", on doit choisir float16 sur T4
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True, raising=False)
    _, dtype = gip.resolve_device_dtype(SimpleNamespace(device="cuda", dtype="auto"))
    assert dtype == torch.float16


def test_resolve_explicit_float32_on_cuda_is_kept(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gip, "bf16_native", lambda: True)
    _, dtype = gip.resolve_device_dtype(SimpleNamespace(device="cuda", dtype="float32"))
    assert dtype == torch.float32


# --------------------------------------------------------------------------- #
# Petits utilitaires
# --------------------------------------------------------------------------- #
def test_mb_conversion():
    assert gip.mb(1024**2) == 1.0
    assert gip.mb(0) == 0.0


def test_sync_and_cleanup_do_not_crash_on_cpu():
    assert gip.sync("cpu") is None
    assert gip.cleanup() is None


def test_make_gen_kwargs_fixed_length():
    kwargs = gip.make_gen_kwargs(FakeTokenizer(), 5, fixed_length=True)
    assert kwargs == {"max_new_tokens": 5, "min_new_tokens": 5, "do_sample": False, "pad_token_id": 0}


def test_make_gen_kwargs_early_stop_allowed():
    kwargs = gip.make_gen_kwargs(FakeTokenizer(), 5, fixed_length=False)
    assert "min_new_tokens" not in kwargs
    assert kwargs["max_new_tokens"] == 5
    assert kwargs["do_sample"] is False


# --------------------------------------------------------------------------- #
# build_inputs
# --------------------------------------------------------------------------- #
def test_build_inputs_uses_chat_template_by_default():
    tok = FakeTokenizer(chat_template="tpl")
    inputs = gip.build_inputs(tok, "salut", batch_size=3, use_chat_template=True, device="cpu")
    assert tok.last_text == ["<chat>salut"] * 3
    assert tok.last_add_special is False  # le template ajoute déjà les tokens spéciaux
    assert inputs["input_ids"].shape[0] == 3


def test_build_inputs_raw_prompt_when_template_disabled():
    tok = FakeTokenizer(chat_template="tpl")
    gip.build_inputs(tok, "salut", batch_size=2, use_chat_template=False, device="cpu")
    assert tok.last_text == ["salut", "salut"]
    assert tok.last_add_special is True


def test_build_inputs_raw_prompt_when_tokenizer_has_no_template():
    tok = FakeTokenizer(chat_template=None)
    gip.build_inputs(tok, "salut", batch_size=1, use_chat_template=True, device="cpu")
    assert tok.last_text == ["salut"]
    assert tok.last_add_special is True


# --------------------------------------------------------------------------- #
# Mesure
# --------------------------------------------------------------------------- #
def test_measure_run_fixed_length_with_fake_model():
    model, tok = FakeModel(), FakeTokenizer()
    args = SimpleNamespace(max_new_tokens=8, allow_early_stop=False)
    r = gip.measure_run(model, tok, make_inputs(batch=2), args, "cpu", run_idx=7)

    # 1ère génération = TTFT (1 token), 2e = génération complète
    assert model.calls == [
        {"max_new_tokens": 1, "min_new_tokens": 1},
        {"max_new_tokens": 8, "min_new_tokens": 8},
    ]
    assert r.run == 7
    assert r.batch_size == 2
    assert r.prompt_tokens == 5
    assert r.new_tokens == 8
    assert r.ttft_ms > 0
    assert r.total_ms > r.ttft_ms
    assert r.decode_ms_per_token > 0
    assert r.decode_tokens_per_s > 0
    assert r.e2e_tokens_per_s > 0
    assert r.peak_vram_mb == 0.0  # pas de GPU


def test_measure_run_early_stop_does_not_force_min_tokens():
    model = FakeModel()
    args = SimpleNamespace(max_new_tokens=4, allow_early_stop=True)
    gip.measure_run(model, FakeTokenizer(), make_inputs(), args, "cpu", run_idx=1)
    assert all(call["min_new_tokens"] is None for call in model.calls)


def test_measure_run_single_token_has_no_decode_phase():
    args = SimpleNamespace(max_new_tokens=1, allow_early_stop=False)
    r = gip.measure_run(FakeModel(), FakeTokenizer(), make_inputs(), args, "cpu", run_idx=1)
    assert r.new_tokens == 1
    assert math.isnan(r.decode_ms_per_token)
    assert math.isnan(r.decode_tokens_per_s)


def test_summarize_statistics():
    s = gip.summarize([1.0, 2.0, 3.0])
    assert s["median"] == 2.0
    assert s["mean"] == 2.0
    assert s["std"] == pytest.approx(1.0)
    assert s["min"] == 1.0
    assert s["max"] == 3.0


def test_summarize_single_value_has_zero_std():
    s = gip.summarize([5.0])
    assert s["std"] == 0.0
    assert s["median"] == 5.0


def test_summarize_runs_has_all_metrics():
    summary = gip.summarize_runs([make_run(1, ttft=10), make_run(2, ttft=20)])
    assert set(summary) == {"ttft_ms", "total_ms", "decode_ms_per_token",
                            "decode_tokens_per_s", "e2e_tokens_per_s", "peak_vram_mb"}
    assert summary["ttft_ms"]["median"] == 15.0
    assert summary["ttft_ms"]["min"] == 10.0
    assert summary["ttft_ms"]["max"] == 20.0


# --------------------------------------------------------------------------- #
# Export et rapport
# --------------------------------------------------------------------------- #
def test_collect_meta_contents(tmp_path):
    args = cpu_args(tmp_path, "--seed", "123", "--batch-size", "4")
    meta = gip.collect_meta(args, "cpu", torch.float32, 0.0, prompt_tokens=12)
    assert meta["device"] == "cpu"
    assert meta["dtype"] == "float32"
    assert meta["batch_size"] == 4
    assert meta["seed"] == 123
    assert meta["prompt_tokens"] == 12
    assert meta["fixed_length"] is True
    assert meta["chat_template"] is True
    assert meta["torch"] == torch.__version__
    assert "gpu" not in meta


def test_save_results_writes_json_and_csv(tmp_path):
    runs = [make_run(1), make_run(2), make_run(3)]
    meta = {"model": "x", "device": "cpu"}
    summary = gip.summarize_runs(runs)
    gip.save_results(tmp_path, meta, runs, summary)

    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data["meta"] == meta
    assert len(data["runs"]) == 3
    assert "ttft_ms" in data["summary"]

    with open(tmp_path / "runs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["run"] == "1"
    assert float(rows[0]["ttft_ms"]) == 10.0


def test_plot_results_creates_png(tmp_path):
    pytest.importorskip("matplotlib")
    gip.plot_results(tmp_path, [make_run(1), make_run(2)], "test")
    png = tmp_path / "benchmark.png"
    assert png.exists() and png.stat().st_size > 0


def test_print_report_shows_all_metrics(tmp_path, capsys):
    args = cpu_args(tmp_path)
    runs = [make_run(1), make_run(2)]
    meta = gip.collect_meta(args, "cpu", torch.float32, 0.0, prompt_tokens=5)
    gip.print_report(meta, gip.summarize_runs(runs))
    out = capsys.readouterr().out
    assert "RÉSULTATS DU BENCHMARK" in out
    for label in ("TTFT", "Latence/token", "Débit décodage", "Pic VRAM"):
        assert label in out


# --------------------------------------------------------------------------- #
# run_benchmark de bout en bout, avec un faux modèle (CPU, sans réseau)
# --------------------------------------------------------------------------- #
def patch_loader(monkeypatch, tokenizer=None, model=None):
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeModel()
    monkeypatch.setattr(gip, "load_model", lambda model_id, device, dtype: (tokenizer, model))
    return tokenizer, model


def test_run_benchmark_end_to_end_with_fake_model(tmp_path, monkeypatch):
    _, model = patch_loader(monkeypatch)
    code = gip.run_benchmark(cpu_args(tmp_path))
    assert code == gip.EXIT_OK

    data = read_results(tmp_path)
    assert len(data["runs"]) == 2
    assert all(r["new_tokens"] == 8 for r in data["runs"])
    assert data["meta"]["dtype"] == "float32"
    assert data["meta"]["warmup"] == 1

    # 1 chauffe + 2 mesures x (TTFT + génération complète) = 5 appels
    assert len(model.calls) == 1 + 2 * 2
    assert len(list(tmp_path.glob("*/runs.csv"))) == 1


def test_run_benchmark_reads_prompt_from_file(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  bonjour granite  \n", encoding="utf-8")
    tok, _ = patch_loader(monkeypatch)
    code = gip.run_benchmark(cpu_args(tmp_path, "--prompt-file", str(prompt_file)))
    assert code == gip.EXIT_OK
    assert tok.last_text == ["<chat>bonjour granite"]


def test_run_benchmark_batch_size_is_propagated(tmp_path, monkeypatch):
    tok, _ = patch_loader(monkeypatch)
    assert gip.run_benchmark(cpu_args(tmp_path, "--batch-size", "3")) == gip.EXIT_OK
    assert len(tok.last_text) == 3
    assert all(r["batch_size"] == 3 for r in read_results(tmp_path)["runs"])


def test_run_benchmark_profile_exports_trace(tmp_path, monkeypatch):
    patch_loader(monkeypatch)
    code = gip.run_benchmark(cpu_args(tmp_path, "--profile", "--profile-tokens", "4"))
    assert code == gip.EXIT_OK
    traces = list(tmp_path.glob("*/trace.json"))
    assert len(traces) == 1 and traces[0].stat().st_size > 0


def test_run_benchmark_returns_load_error_code(tmp_path, monkeypatch, capsys):
    def boom(model_id, device, dtype):
        raise OSError("modèle introuvable")

    monkeypatch.setattr(gip, "load_model", boom)
    assert gip.run_benchmark(cpu_args(tmp_path)) == gip.EXIT_LOAD_ERROR
    assert "ERREUR de chargement" in capsys.readouterr().err


def test_run_benchmark_returns_oom_code(tmp_path, monkeypatch, capsys):
    def boom(model_id, device, dtype):
        raise torch.cuda.OutOfMemoryError("oom simulé")

    monkeypatch.setattr(gip, "load_model", boom)
    assert gip.run_benchmark(cpu_args(tmp_path)) == gip.EXIT_OOM
    assert "OOM" in capsys.readouterr().err


def test_cli_help_runs():
    res = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0
    assert "--max-new-tokens" in res.stdout
    assert "--profile" in res.stdout


# --------------------------------------------------------------------------- #
# Intégration : vrai pipeline Hugging Face sur un mini-modèle (réseau requis)
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_tiny_model_end_to_end(tmp_path):
    args = gip.parse_args([
        "--model", TINY_MODEL, "--output-dir", str(tmp_path),
        "--warmup", "1", "--repeats", "2", "--max-new-tokens", "8", "--no-plots",
    ])
    assert gip.run_benchmark(args) == gip.EXIT_OK
    data = read_results(tmp_path)
    assert len(data["runs"]) == 2
    assert all(r["new_tokens"] == 8 for r in data["runs"])
    assert data["summary"]["ttft_ms"]["median"] > 0


@pytest.mark.integration
def test_tiny_model_batch_and_profile(tmp_path):
    args = gip.parse_args([
        "--model", TINY_MODEL, "--output-dir", str(tmp_path), "--batch-size", "3",
        "--warmup", "1", "--repeats", "2", "--max-new-tokens", "6", "--no-plots",
        "--profile", "--profile-tokens", "4",
    ])
    assert gip.run_benchmark(args) == gip.EXIT_OK
    data = read_results(tmp_path)
    assert all(r["batch_size"] == 3 for r in data["runs"])
    assert len(list(tmp_path.glob("*/trace.json"))) == 1


@pytest.mark.integration
def test_unknown_model_returns_load_error(tmp_path):
    args = gip.parse_args([
        "--model", "ce-modele/nexiste-pas-du-tout-12345", "--output-dir", str(tmp_path),
        "--warmup", "0", "--repeats", "1", "--max-new-tokens", "4", "--no-plots",
    ])
    assert gip.run_benchmark(args) == gip.EXIT_LOAD_ERROR


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="GPU CUDA requis")
def test_gpu_run_reports_memory_and_gpu_metadata(tmp_path):
    args = gip.parse_args([
        "--model", TINY_MODEL, "--device", "cuda", "--dtype", "float16", "--output-dir", str(tmp_path),
        "--warmup", "1", "--repeats", "2", "--max-new-tokens", "8", "--no-plots",
    ])
    assert gip.run_benchmark(args) == gip.EXIT_OK
    data = read_results(tmp_path)
    meta, summary = data["meta"], data["summary"]
    assert meta["device"] == "cuda"
    assert meta["dtype"] == "float16"
    assert meta["gpu"]
    assert meta["vram_model_mb"] > 0
    assert summary["peak_vram_mb"]["median"] >= meta["vram_model_mb"]


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="GPU CUDA requis")
def test_gpu_timing_is_synchronized():
    """Sans synchronize(), un kernel asynchrone paraîtrait quasi instantané."""
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        x = x @ x
        x = x / x.norm()
    gip.sync("cuda")
    synced = time.perf_counter() - t0
    assert synced > 0
    assert torch.isfinite(x).all()


# --------------------------------------------------------------------------- #
# Vrai modèle Granite (long : RUN_GRANITE=1 et GPU requis)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_GRANITE") != "1" or not HAS_CUDA,
                    reason="RUN_GRANITE=1 et GPU requis")
def test_real_granite_on_gpu(tmp_path):
    args = gip.parse_args([
        "--output-dir", str(tmp_path), "--warmup", "1", "--repeats", "2",
        "--max-new-tokens", "32", "--no-plots",
    ])
    assert gip.run_benchmark(args) == gip.EXIT_OK
    data = read_results(tmp_path)
    assert data["summary"]["decode_tokens_per_s"]["median"] > 1
    assert data["meta"]["vram_model_mb"] > 1000

    # Le modèle doit être libéré à la fin du run (sinon on ne peut pas enchaîner plusieurs modèles)
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_reserved() < 500 * 1024**2
