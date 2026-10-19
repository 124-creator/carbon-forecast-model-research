"""Rerun an unchanged original audit, redirecting its receipt only.

No numerical fixture, scientific source or original receipt is modified.
Run each audit in a separate process because the runner audit checks imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", choices=("core", "runner"))
    args = parser.parse_args()
    script = ROOT / "audits" / f"independent_{args.audit}_audit.py"
    original_name = ("INDEPENDENT_CORE_AUDIT.json" if args.audit == "core"
                     else "RUNNER_PREFIT_AUDIT.json")
    new_name = ("INDEPENDENT_CORE_AUDIT_RECOVERY.json" if args.audit == "core"
                else "RUNNER_PREFIT_AUDIT_RECOVERY.json")
    original_receipt = ROOT / "audits" / original_name
    before = {str(script.relative_to(ROOT)): sha(script),
              str(original_receipt.relative_to(ROOT)): sha(original_receipt)}
    text = script.read_text()
    needle = f'"{original_name}"'
    assert text.count(needle) == 1, "Only the single receipt filename may change"
    redirected = text.replace(needle, f'"{new_name}"')
    namespace = {"__name__": "__main__", "__file__": str(script),
                 "__package__": None}
    status = 0
    try:
        exec(compile(redirected, str(script), "exec"), namespace)
    except SystemExit as exc:
        status = int(exc.code or 0)
    after = {relative: sha(ROOT / relative) for relative in before}
    assert before == after, "Original audit or historical receipt changed"
    output = ROOT / "audits" / new_name
    receipt = json.loads(output.read_text())
    assert receipt["audit_script_sha256"] == before[str(script.relative_to(ROOT))]
    wrapper_receipt = {
        "audit": args.audit, "exit_status": status,
        "output_receipt": str(output.relative_to(ROOT)),
        "original_audit_script_sha256": sha(script),
        "wrapper_sha256": sha(Path(__file__)),
        "source_text_change": "Only output receipt filename redirected in memory",
        "original_script_and_receipt_unchanged": before == after,
        "original_files": before,
        "output_receipt_sha256": sha(output),
        "numerical_fixtures_and_tolerances_changed": False,
    }
    (ROOT / "audits" / f"RECOVERY_{args.audit.upper()}_AUDIT_WRAPPER.json").write_text(
        json.dumps(wrapper_receipt, indent=2, ensure_ascii=False) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
