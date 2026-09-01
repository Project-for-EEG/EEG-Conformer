"""
Parse the Siena Scalp EEG annotations into machine-readable seizure times.

The Siena "Seizures-list-PNxx.txt" files are free text written by hand, and
they are inconsistent in ways that break any naive parser:

  * PN06 lists its files as PNO6-1.edf -- capital letter O, not zero
  * PN11 lists PN11-.edf, with the index missing
  * PN12 seizure 2 gives no registration time, inheriting seizure 1's
  * PN00-3 ends its seizure at 19.29.29 when the recording stops at 18.57.13,
    an hour after the recording ended
  * PN10-2 ends with "11.41.04 opure 11.40.43"
  * PN10-3 gives two onsets, clinical and electric
  * PN12 writes one timestamp as 16:13.23, colon instead of dot
  * recordings cross midnight, so 21.11.29 -> 00.41.49 is 3.5 h, not -20.5 h

Rather than trust the text, seizure offsets are computed against the start
time in the EDF *header*, which is machine-written. The registration times in
the text are only used as a cross-check, and disagreements are reported.

Output: siena_seizures.json, one record per seizure that has a real file.
"""
import json
import re
from pathlib import Path

import mne

SIENA = Path("additional_data/siena")
DAY = 24 * 3600


def to_seconds(text):
    """21.51.02, 21:51:02 and 16:13.23 all mean the same thing."""
    parts = re.split(r"[.:]", text.strip())
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
    except ValueError:
        return None
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        return None
    return h * 3600 + m * 60 + s


def first_time(line):
    """First clock time on the line, ignoring trailing commentary such as
    (CLINICAL ONSET); 15.43.59 (ELECTRIC ONSET), or the stray "opure"."""
    m = re.search(r"(\d{1,2}[.:]\d{2}[.:]\d{2})", line)
    return to_seconds(m.group(1)) if m else None


def normalise_filename(raw, patient, available):
    """Map the name written in the text onto a file that exists.

    Handles the letter-O typo, the missing index in PN11-.edf, and the
    multi-seizure names such as PN10-4.5.6.edf.
    """
    name = raw.strip().replace("PNO", "PN0").replace("PNo", "PN0")
    if name in available:
        return name
    stem = name[:-4] if name.lower().endswith(".edf") else name
    if stem.endswith("-"):                      # PN11- -> PN11-1
        cand = stem + "1.edf"
        if cand in available:
            return cand
    matches = [f for f in available if f.lower() == name.lower()]
    return matches[0] if matches else None


def parse_patient(patient):
    txt = (SIENA / patient / ("Seizures-list-" + patient + ".txt")).read_text(
        errors="replace").replace("\r", "")
    available = sorted(p.name for p in (SIENA / patient).glob("*.edf"))

    # split into seizure blocks; "Seizure n 1", "Seizure n1", "Seizure n 1:"
    blocks = re.split(r"\n(?=\s*Seizure\s*n\s*\d)", txt)
    records, carried_file = [], None

    for block in blocks:
        if not re.match(r"\s*Seizure\s*n\s*\d", block):
            continue
        idx = int(re.search(r"Seizure\s*n\s*(\d+)", block).group(1))

        m = re.search(r"File name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if m:
            fname = normalise_filename(m.group(1), patient, available)
            carried_file = fname or carried_file
        else:
            # PN12 seizure 2 continues in the previous block's file; PN01
            # names no file at all and has only one recording
            fname = carried_file or (available[0] if len(available) == 1 else None)

        start = end = reg_start = None
        for line in block.split("\n"):
            low = line.lower()
            if "seizure start" in low or re.match(r"\s*start time", low):
                start = first_time(line)
            elif "seizure end" in low or re.match(r"\s*end time", low):
                end = first_time(line)
            elif "registration start" in low:
                reg_start = first_time(line)

        records.append({"patient": patient, "index": idx, "file": fname,
                        "clock_start": start, "clock_end": end,
                        "clock_reg_start": reg_start})
    return records


def main():
    patients = sorted(p.name for p in SIENA.iterdir() if p.is_dir())
    all_records, skipped = [], []

    for patient in patients:
        for rec in parse_patient(patient):
            if rec["file"] is None:
                skipped.append((rec["patient"], rec["index"], "no matching EDF"))
                continue
            path = SIENA / patient / rec["file"]

            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            meas = raw.info["meas_date"]
            edf_start = meas.hour * 3600 + meas.minute * 60 + meas.second
            duration = raw.n_times / raw.info["sfreq"]

            if rec["clock_start"] is None or rec["clock_end"] is None:
                skipped.append((patient, rec["index"], "unparseable times"))
                continue

            # wall clock to offset from the start of the recording, wrapping
            # at midnight so overnight files come out positive
            onset = (rec["clock_start"] - edf_start) % DAY
            offset = (rec["clock_end"] - edf_start) % DAY

            if onset > duration:
                skipped.append((patient, rec["index"],
                                "onset %.0fs beyond file %.0fs"
                                % (onset, duration)))
                continue

            note = ""
            if offset <= onset or offset > duration:
                # PN00-3: the end time is written an hour late. A seizure that
                # ends before it starts, or after the recording stops, is a
                # typo, not a finding -- fall back to a typical Siena length.
                fixed = min(onset + 60.0, duration)
                note = ("end time invalid (offset %.0fs vs %.0fs file); "
                        "replaced with onset + %.0f s"
                        % (offset, duration, fixed - onset))
                offset = fixed

            drift = None
            if rec["clock_reg_start"] is not None:
                drift = (rec["clock_reg_start"] - edf_start + DAY // 2) % DAY - DAY // 2

            all_records.append({
                "patient": patient, "index": rec["index"], "file": rec["file"],
                "onset_s": round(onset, 3), "offset_s": round(offset, 3),
                "duration_s": round(offset - onset, 3),
                "file_duration_s": round(duration, 1),
                "sfreq": raw.info["sfreq"], "n_channels": len(raw.ch_names),
                "edf_start_clock": "%02d.%02d.%02d" % (meas.hour, meas.minute,
                                                       meas.second),
                "text_vs_header_drift_s": drift, "note": note})

    n_pat = len(set(r["patient"] for r in all_records))
    print("%d seizures parsed across %d patients\n" % (len(all_records), n_pat))
    print("%-8s %2s %-18s %9s %6s %9s %6s"
          % ("patient", "#", "file", "onset", "dur", "file_len", "drift"))
    for r in all_records:
        d = "" if r["text_vs_header_drift_s"] is None else "%+.0f" % r["text_vs_header_drift_s"]
        print("%-8s %2d %-18s %9.1f %6.1f %9.1f %6s"
              % (r["patient"], r["index"], r["file"], r["onset_s"],
                 r["duration_s"], r["file_duration_s"], d))
        if r["note"]:
            print("         ^ " + r["note"])

    if skipped:
        print("\n%d listed seizures not usable:" % len(skipped))
        for p, i, why in skipped:
            print("  %s seizure %d: %s" % (p, i, why))

    total = sum(r["duration_s"] for r in all_records)
    hours = sum(dict(((r["patient"], r["file"]), r["file_duration_s"])
                     for r in all_records).values()) / 3600
    print("\n%.0f s of seizure in %.1f h of seizure-bearing recording (%.3f%%)"
          % (total, hours, 100 * total / (hours * 3600)))

    Path("siena_seizures.json").write_text(json.dumps(all_records, indent=2))
    print("wrote siena_seizures.json")


if __name__ == "__main__":
    main()
