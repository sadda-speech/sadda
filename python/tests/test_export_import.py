"""Python-surface tests for CSV / JSON annotation export + import
(`Project.export_csv` / `export_json` / `import_csv` / `import_json`).

The round-trip exports a bundle's annotations and re-imports them into a
*second* bundle, asserting the honoured fields survive. Per the v1 import
limits, `status` / `parent_annotation_id` / `processing_run_id` are dropped;
times, label, note, and extra are honoured."""

from __future__ import annotations

import json
import tempfile
import wave
from pathlib import Path

import sadda


def _write_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * (sample_rate // 4))


def _project_with_two_bundles(td: Path) -> tuple[sadda.Project, int, int]:
    proj = sadda.new_project(td / "p", "demo")
    # Distinct source files: add_bundle copies the WAV into the project keyed
    # by filename, so two bundles can't share one source path.
    src_wav, dst_wav = td / "src.wav", td / "dst.wav"
    _write_wav(src_wav)
    _write_wav(dst_wav)
    src = proj.add_bundle("src", src_wav)
    dst = proj.add_bundle("dst", dst_wav)
    return proj, src, dst


def _annotate_source(proj: sadda.Project, bundle_id: int) -> None:
    words = proj.add_tier(bundle_id, "words", "interval")
    proj.add_interval(words, 0.0, 0.5, label="hello")
    # A label with the CSV-hostile trio: comma, quote, newline.
    proj.add_interval(words, 0.5, 1.0, label='a, b "c"\nd')
    pulses = proj.add_tier(bundle_id, "pulses", "point")
    proj.add_point(pulses, 0.25, label="p")


def test_export_csv_then_import_round_trips() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, src, dst = _project_with_two_bundles(Path(td))
        _annotate_source(proj, src)

        csv_path = Path(td) / "annotations.csv"
        proj.export_csv(src, csv_path)
        assert csv_path.exists()
        header = csv_path.read_text().splitlines()[0]
        assert header.startswith("bundle_id,bundle_name,tier_id,tier_name,tier_type")

        new_tiers = proj.import_csv(csv_path, dst)
        assert len(new_tiers) == 2

        tiers = {t.name: t for t in proj.tiers(dst)}
        assert set(tiers) == {"words", "pulses"}
        assert tiers["words"].type == "interval"
        assert tiers["pulses"].type == "point"

        ivs = proj.intervals(tiers["words"].id)
        assert [(round(i.start_seconds, 3), round(i.end_seconds, 3)) for i in ivs] == [
            (0.0, 0.5),
            (0.5, 1.0),
        ]
        # The hostile label survives the quote round-trip.
        assert ivs[1].label == 'a, b "c"\nd'

        pts = proj.points(tiers["pulses"].id)
        assert len(pts) == 1
        assert round(pts[0].time_seconds, 3) == 0.25
        assert pts[0].label == "p"


def test_export_json_is_structured_and_imports() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, src, dst = _project_with_two_bundles(Path(td))
        _annotate_source(proj, src)

        json_path = Path(td) / "annotations.json"
        proj.export_json(src, json_path)

        doc = json.loads(json_path.read_text())
        assert doc["bundle"]["name"] == "src"
        assert doc["bundle"]["duration_seconds"] > 0
        names = {t["name"]: t for t in doc["tiers"]}
        assert names["words"]["type"] == "interval"
        assert names["words"]["intervals"][0]["label"] == "hello"
        assert names["pulses"]["type"] == "point"

        new_tiers = proj.import_json(json_path, dst)
        assert len(new_tiers) == 2
        tiers = {t.name: t for t in proj.tiers(dst)}
        assert [i.label for i in proj.intervals(tiers["words"].id)] == [
            "hello",
            'a, b "c"\nd',
        ]


def test_export_json_embeds_extra_as_object() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, src, _dst = _project_with_two_bundles(Path(td))
        tier = proj.add_tier(src, "words", "interval")
        proj.add_interval(tier, 0.0, 0.5, label="hi", extra='{"f0": 120.5}')

        json_path = Path(td) / "a.json"
        proj.export_json(src, json_path)
        doc = json.loads(json_path.read_text())
        # extra is embedded as a nested object, not an escaped string.
        assert doc["tiers"][0]["intervals"][0]["extra"] == {"f0": 120.5}
