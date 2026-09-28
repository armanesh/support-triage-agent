from triageagent.cli import main


def test_run_processes_sample_tickets(tmp_path, capsys):
    queue_path = tmp_path / "escalations.jsonl"
    rc = main(["run", "--queue", str(queue_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Processed 10 tickets" in captured.out
    assert queue_path.exists()


def test_handle_single_ticket(tmp_path, capsys):
    queue_path = tmp_path / "escalations.jsonl"
    rc = main(["handle", "please refund ORD-1001", "--subject", "refund", "--queue", str(queue_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "category=billing_refund" in captured.out


def test_show_queue_empty(tmp_path, capsys):
    queue_path = tmp_path / "escalations.jsonl"
    rc = main(["show-queue", "--queue", str(queue_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "empty" in captured.out.lower()


def test_show_queue_after_escalation(tmp_path, capsys):
    queue_path = tmp_path / "escalations.jsonl"
    main(["handle", "I am furious this is a scam", "--subject", "angry", "--queue", str(queue_path)])
    capsys.readouterr()

    rc = main(["show-queue", "--queue", str(queue_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "reason:" in captured.out
