import shutil
import subprocess
from pathlib import Path

import pytest

from core import audio_utils as au
from core.cli import main as cli_main
from core.dataset_builder import read_metadata_jsonl
from core.events import Stage, overall_percent
from tests.synth import TrueRateAligner, make_text, synth_reading
from workers.pipeline_runner import KIND_DATASET, KIND_LORA, TaskRequest, plan_for, run_task


def _inputs(tmp_path, n=40):
    text = make_text(n, 4)
    audio, _ = synth_reading(text)
    wav, txt = tmp_path / "Мой голос.wav", tmp_path / "t.txt"
    au.write_wav(wav, audio, 16000)
    txt.write_text(text, encoding="utf-8")
    return wav, txt


class NotDueUpdater:
    def should_autocheck(self):
        return False


def test_run_task_dataset_and_default_folder(tmp_path):
    wav, txt = _inputs(tmp_path)
    stages = []
    res = run_task(TaskRequest(KIND_DATASET, wav, txt), lambda s, f, m: stages.append(s),
                   updater=NotDueUpdater(), aligner_factory=TrueRateAligner)
    assert res.root_dir.parent == tmp_path and res.root_dir.name.endswith("_Voxprint")
    assert (res.dataset_dir / "metadata.jsonl").exists() and res.adapter_path is None
    assert len(read_metadata_jsonl(res.dataset_dir / "metadata.jsonl")) == res.n_segments
    assert not (res.dataset_dir / "train_24k").exists()
    assert set(stages) <= set(plan_for(KIND_DATASET))


def test_run_task_lora_passes_language_and_warnings_to_trainer(tmp_path, monkeypatch):
    wav, txt = _inputs(tmp_path)
    called = {}

    def fake_train(dataset_dir, output_dir, progress, cancel, force_cpu, language=None, warnings_out=None):
        called.update(d=dataset_dir, o=output_dir, cpu=force_cpu, lang=language)
        warnings_out.append("предупреждение обучения")
        return output_dir

    import core.lora_trainer as lt
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    stages = []
    res = run_task(TaskRequest(KIND_LORA, wav, txt, out_root=tmp_path / "out", force_cpu=True),
                   lambda s, f, m: stages.append(s), updater=NotDueUpdater(), aligner_factory=TrueRateAligner)
    assert (res.dataset_dir / "ref.wav").exists() and (res.dataset_dir / "ref_text.txt").exists()
    assert not (res.dataset_dir / "train_24k").exists()
    assert called["cpu"] is True and res.adapter_path == tmp_path / "out" / "output"
    assert called["lang"] == "russian" and "предупреждение обучения" in res.warnings
    assert called["o"] == tmp_path / "out" / "output"
    assert Stage.SAVE not in stages  # SAVE показывает только тренер; запись датасета идёт под «Нарезка»


def test_overall_percent_monotonic_over_plan():
    plan = plan_for(KIND_LORA)
    vals = [overall_percent(plan, s, f) for s in plan for f in (0, 0.5, 1)]
    assert vals == sorted(vals) and vals[0] == 0 and vals[-1] == 100


def test_cli_fake_aligner(tmp_path, capsys):
    wav, txt = _inputs(tmp_path, 25)
    assert cli_main([str(wav), str(txt), "--out", str(tmp_path / "ds"), "--fake-aligner"]) == 0
    assert (tmp_path / "ds" / "metadata.jsonl").exists()
    assert "Готово" in capsys.readouterr().out


def test_cli_friendly_error(tmp_path, capsys):
    t = tmp_path / "t.txt"
    t.write_text("текст", encoding="utf-8")
    assert cli_main([str(tmp_path / "nope.wav"), str(t), "--out", str(tmp_path / "o"), "--fake-aligner"]) == 2
    assert "ОШИБКА" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")
def test_mp3_via_pydub_ffmpeg(tmp_path):
    text = "Раз два три. Четыре пять шесть. Семь восемь девять."
    audio, _ = synth_reading(text)
    wav = tmp_path / "a.wav"
    au.write_wav(wav, audio, 16000)
    mp3 = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), str(mp3)], check=True)
    x, sr = au.load_audio(mp3, 16000)
    assert sr == 16000 and abs(len(x) / 16000 - len(audio) / 16000) < 0.2


def test_prefetch_downloads_only_missing(tmp_path, monkeypatch):
    from infra import model_downloader as md
    from workers.pipeline_runner import models_missing, prefetch_models
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "h"))
    got = []

    def ensure(repo, progress):
        got.append(repo)
        d = md.local_dir_for(repo)
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "m.safetensors").write_bytes(b"0")
        progress(Stage.MODEL, 1.0, "ok")

    repos = ["A/a", "B/b"]
    assert models_missing(repos) == repos
    assert prefetch_models(repos=repos, ensure=ensure) == repos
    assert prefetch_models(repos=repos, ensure=ensure) == [] and got == repos
