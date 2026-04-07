#!/usr/bin/env python3
"""Stream-split sweep_trades.json (one giant dict) into per-config files.

The JSON has shape {"label1":[...],"label2":[...],...}. We scan it linearly,
tracking string-state and bracket depth, and emit one file per top-level entry.
This avoids ever loading the full 4 GB into memory.
"""
import json
import os
import sys

IN = "/mnt/c/Users/cr0wn/fvg-intelligence/scripts/sol_sweep_results/sweep_trades.json"
OUT_DIR = "/mnt/c/Users/cr0wn/fvg-intelligence/scripts/sol_sweep_results/per_config"
CHUNK = 4 * 1024 * 1024


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    f = open(IN, "rb")
    buf = bytearray()
    pos = 0  # absolute byte index in stream consumed
    state = "expect_open"
    in_str = False
    esc = False
    depth = 0
    key = None
    value_start = None
    written = 0

    def refill():
        nonlocal buf
        chunk = f.read(CHUNK)
        if not chunk:
            return False
        buf.extend(chunk)
        return True

    # Read until first '{'
    while True:
        if not buf:
            if not refill():
                return
        c = buf[0:1]
        if c == b"{":
            del buf[0]
            state = "expect_key"
            break
        else:
            del buf[0]

    while True:
        # Ensure we have data
        if not buf:
            if not refill():
                break

        if state == "expect_key":
            # Skip whitespace and commas
            while buf and buf[0:1] in (b" ", b"\n", b"\t", b"\r", b","):
                del buf[0]
            if not buf:
                if not refill():
                    break
                continue
            if buf[0:1] == b"}":
                break
            if buf[0:1] != b'"':
                # Should not happen
                print(f"unexpected char: {buf[0:1]}")
                return
            # Parse the key string
            del buf[0]  # consume opening quote
            key_bytes = bytearray()
            while True:
                if not buf:
                    if not refill():
                        return
                ch = buf[0:1]
                del buf[0]
                if ch == b"\\":
                    if not buf:
                        if not refill():
                            return
                    key_bytes += ch + buf[0:1]
                    del buf[0]
                    continue
                if ch == b'"':
                    break
                key_bytes += ch
            key = key_bytes.decode("utf-8")
            state = "expect_colon"

        elif state == "expect_colon":
            while buf and buf[0:1] in (b" ", b"\n", b"\t", b"\r"):
                del buf[0]
            if not buf:
                if not refill():
                    return
                continue
            if buf[0:1] == b":":
                del buf[0]
                state = "expect_value"

        elif state == "expect_value":
            while buf and buf[0:1] in (b" ", b"\n", b"\t", b"\r"):
                del buf[0]
            if not buf:
                if not refill():
                    return
                continue
            # Now scan for matching close. Could be array, object, scalar.
            depth = 0
            in_str = False
            esc = False
            value_bytes = bytearray()
            started = False
            while True:
                if not buf:
                    if not refill():
                        # End of file mid-value
                        break
                ch = buf[0:1]
                value_bytes += ch
                del buf[0]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == b"\\":
                        esc = True
                    elif ch == b'"':
                        in_str = False
                    continue
                if ch == b'"':
                    in_str = True
                    continue
                if ch in (b"[", b"{"):
                    depth += 1
                    started = True
                elif ch in (b"]", b"}"):
                    depth -= 1
                    if depth == 0 and started:
                        break
                elif depth == 0 and started is False:
                    # scalar value (number/bool/null) ends at , or }
                    pass

            # Write out
            out_path = os.path.join(OUT_DIR, f"{key}.json")
            if not os.path.exists(out_path):
                with open(out_path, "wb") as out:
                    out.write(value_bytes)
            written += 1
            if written % 20 == 0:
                print(f"  {written} configs written", flush=True)
            state = "expect_key"

    print(f"Done: {written} configs in {OUT_DIR}")


if __name__ == "__main__":
    main()
