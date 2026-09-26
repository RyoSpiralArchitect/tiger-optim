#!/usr/bin/env python3
"""Group scalar host reads by enclosing PyTorch operators in a CPU trace."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    by_thread = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and "dur" in event and "ts" in event:
            by_thread[event.get("tid")].append(event)
    chains = Counter()
    for thread_events in by_thread.values():
        thread_events.sort(key=lambda event: (event["ts"], -event["dur"]))
        stack = []
        for event in thread_events:
            start = event["ts"]
            end = start + event["dur"]
            while stack and start >= stack[-1]["ts"] + stack[-1]["dur"] - 1e-8:
                stack.pop()
            while stack and end > stack[-1]["ts"] + stack[-1]["dur"] + 1e-8:
                stack.pop()
            if event["name"] == "aten::_local_scalar_dense":
                chain = [parent["name"] for parent in stack
                         if parent["name"].startswith("aten::")]
                chains[" > ".join(chain)] += 1
            stack.append(event)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(chains.most_common()), indent=2) + "\n")
    print(json.dumps(dict(chains.most_common()), sort_keys=True))


if __name__ == "__main__":
    main()
