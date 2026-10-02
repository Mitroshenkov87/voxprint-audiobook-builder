"""CLI для проверки этапов 1-2 на машине с GPU:

    python -m core.cli audio.wav text.txt --out dataset [--language Russian] [--device auto|cuda|cpu]
                       [--fake-aligner] [--train] [--output-dir output]

Только для разработчиков/тестирования; пользователь работает через графическое окно.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.aligner import FakeAligner, make_default_aligner
from core.dataset_builder import BuildConfig, DatasetBuilder
from core.errors import DatasetMakerError
from core.events import Stage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.cli", description="Voxprint: аудио + текст -> датасет")
    ap.add_argument("audio")
    ap.add_argument("text")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--language", default=None, help="Russian, English, ... (по умолчанию определяется)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--fake-aligner", action="store_true", help="без нейросети (сухой прогон)")
    ap.add_argument("--model", default=None, help="путь/ID модели выравнивателя")
    ap.add_argument("--train", action="store_true", help="после датасета обучить LoRA")
    ap.add_argument("--output-dir", default="output", help="куда сохранить адаптер (с --train)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def progress(stage: Stage, frac: float, msg: str) -> None:
        print(f"[{stage.label}] {frac * 100:5.1f}%  {msg}", flush=True)

    try:
        if args.fake_aligner:
            aligner = FakeAligner()
        else:
            from infra.model_downloader import ensure_aligner_model

            path = args.model or ensure_aligner_model(progress=progress)
            dev = {"cuda": "cuda:0"}.get(args.device, args.device)
            aligner = make_default_aligner(str(path), dev)
        cfg = BuildConfig(sample_rate=args.sr, language=args.language)
        res = DatasetBuilder(aligner, cfg).run(args.audio, args.text, Path(args.out), progress)
        print(f"\nГотово: {res.n_segments} сегментов, {res.total_seconds:.0f} с -> {res.dataset_dir}")
        for w in res.warnings:
            print("  ! " + w)
        if args.train:
            aligner.unload()
            from core.lora_trainer import train_lora_from_dataset

            out = train_lora_from_dataset(res.dataset_dir, Path(args.output_dir), progress=progress,
                                      language=res.training_language)
            print(f"Адаптер: {out}")
        return 0
    except DatasetMakerError as exc:
        print(f"ОШИБКА: {exc.user_message}", file=sys.stderr)
        if exc.details:
            print(f"  детали: {exc.details}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
