import json
import runpy
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from aiops_pipeline import load_data, run_pipeline  # noqa: E402
from anomaly_detector import AnomalyDetector  # noqa: E402
from calculations import area_of_circle, get_nth_fibonacci  # noqa: E402
from event_producer import EventProducer  # noqa: E402
from event_topic import EventTopic  # noqa: E402


def test_calculations_reject_negative_inputs():
    with pytest.raises(ValueError, match="Radius cannot be negative"):
        area_of_circle(-1)

    with pytest.raises(ValueError, match="n cannot be negative"):
        get_nth_fibonacci(-1)


def test_get_nth_fibonacci_iterates_for_larger_input():
    assert get_nth_fibonacci(10) == 55


def test_anomaly_detector_reports_each_threshold_reason():
    detector = AnomalyDetector()
    record = {
        "timestamp": "2026-09-20T10:10:00",
        "service": "payment-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
        "message": "Payment service is under load",
    }

    event = detector.detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]
    assert event["source"] == record


def test_event_producer_rejects_empty_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)

    assert producer.publish(None) is False
    assert topic.get_messages() == []


def test_event_topic_clear_removes_published_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})

    topic.clear()

    assert topic.get_messages() == []


def test_pipeline_loads_data_and_detects_anomalies(tmp_path):
    records = [
        {
            "timestamp": "2026-09-20T10:00:00",
            "service": "payment-service",
            "response_time_ms": 120,
            "cpu_percent": 42,
            "memory_percent": 51,
            "log_level": "INFO",
            "message": "Payment request processed successfully",
        },
        {
            "timestamp": "2026-09-20T10:05:00",
            "service": "payment-service",
            "response_time_ms": 610,
            "cpu_percent": 75,
            "memory_percent": 70,
            "log_level": "ERROR",
            "message": "Payment service timeout",
        },
    ]
    data_file = tmp_path / "service_data.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")

    assert load_data(data_file) == records
    result = run_pipeline(data_file)

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["anomalies_detected"][0]["service"] == "payment-service"
    assert result["events_consumed"] == []


def test_pipeline_command_prints_summary(monkeypatch, capsys):
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repository_root)

    runpy.run_path(str(SRC_DIR / "aiops_pipeline.py"), run_name="__main__")

    output = capsys.readouterr().out
    assert "AIOps Pipeline Result" in output
    assert "Records processed:" in output
    assert "Anomalies detected:" in output
    assert "Events consumed:" in output
