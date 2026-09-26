"""shrutlekh CLI: add, list, show, search, delete."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from shrutlekh import db, store, worker
from shrutlekh.summarize import TEMPLATES


def _when(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def cmd_add(args):
    rid = store.ingest(args.audio, args.mode, args.template, args.title)
    print(f"{rid}  ingested, processing...")
    worker.process(rid, args.tier)
    rec = store.get_recording(rid)
    print(f"{rid}  {rec['status']}  {rec['duration_s']:.0f}s  {rec['tier']}/{Path(rec['asr_model']).name}")
    cmd_show(argparse.Namespace(id=rid, transcript=False))


def cmd_list(args):
    rows = store.list_recordings(args.limit)
    if not rows:
        print("no recordings yet")
        return
    for r in rows:
        dur = f"{r['duration_s']:.0f}s" if r["duration_s"] else "-"
        print(f"{r['id']}  {_when(r['created_at'])}  {r['status']:<10} {r['mode']:<8} {dur:>6}  "
              f"{r['n_segments']:>3} seg  {r['title']}")


def cmd_show(args):
    rec = store.get_recording(args.id)
    if rec is None:
        print(f"no recording {args.id}")
        return
    print(f"{rec['title']}  [{rec['status']}]  {rec['mode']}/{rec['template']}  {_when(rec['created_at'])}")
    if rec["error"]:
        print(f"error: {rec['error']}")
    if args.transcript:
        for s in store.get_segments(args.id):
            print(f"  [{_mmss(s.start)}-{_mmss(s.end)}] {s.text}")
    for n in store.get_notes(args.id):
        print(f"\n--- {n['template']} ({n['model']}, {n['timings'].get('asr_s')}s asr + "
              f"{n['timings'].get('llm_s')}s llm) ---\n{n['summary']}")
        if n["action_items"]:
            print("\naction items:\n" + json.dumps(n["action_items"], ensure_ascii=False, indent=1))


def cmd_search(args):
    hits = store.search(args.query, args.limit)
    if not hits:
        print("no matches")
        return
    for h in hits:
        print(f"{h['recording_id']}  [{_mmss(h['t_start'])}]  {h['title']}\n    {h['snippet']}")


def cmd_delete(args):
    print("deleted" if store.delete_recording(args.id) else f"no recording {args.id}")


p = argparse.ArgumentParser(prog="shrutlekh")
sub = p.add_subparsers(dest="cmd", required=True)

a = sub.add_parser("add", help="ingest an audio file and process it")
a.add_argument("audio", type=Path)
a.add_argument("--mode", choices=("hindi", "hinglish"), default="hindi")
a.add_argument("--template", choices=tuple(TEMPLATES), default="meeting")
a.add_argument("--tier", choices=("cpu", "gpu"))
a.add_argument("--title")
a.set_defaults(func=cmd_add)

ls = sub.add_parser("list", help="list recordings, newest first")
ls.add_argument("--limit", type=int, default=50)
ls.set_defaults(func=cmd_list)

sh = sub.add_parser("show", help="show a recording's notes")
sh.add_argument("id")
sh.add_argument("--transcript", action="store_true")
sh.set_defaults(func=cmd_show)

se = sub.add_parser("search", help="keyword search across transcripts")
se.add_argument("query")
se.add_argument("--limit", type=int, default=20)
se.set_defaults(func=cmd_search)

de = sub.add_parser("delete", help="delete a recording and its notes")
de.add_argument("id")
de.set_defaults(func=cmd_delete)

args = p.parse_args()
db.init_db()
args.func(args)
