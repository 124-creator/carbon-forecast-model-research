"""Portable launcher preserving the original audit scripts and receipts."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",choices=("events","targets","core","logs"),required=True)
    ap.add_argument("--study",type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--previous-study",type=Path)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    output=args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; original receipts are preserved.")
    output.mkdir(parents=True,exist_ok=True)
    scripts=dict(events="event_information_synthetic.py",targets="dsof_target_synthetic.py",
                 core="audit_rrc_core.py",logs="log_diagnostics.py")
    source=Path(__file__).resolve().parent/scripts[args.case]
    spec=importlib.util.spec_from_file_location("portable_audit_"+args.case,source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.HERE=output
    if args.case=="core":
        module.SOURCE=args.study.resolve()/"source/rrc_core.py"
    if args.case=="logs":
        if args.previous_study is None:
            raise ValueError("--previous-study is required for old public receipt diagnostics.")
        module.ROOT=args.previous_study.resolve()
    module.main()


if __name__=="__main__":
    main()
