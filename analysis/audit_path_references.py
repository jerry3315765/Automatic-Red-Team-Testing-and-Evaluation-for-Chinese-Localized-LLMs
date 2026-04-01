import argparse
import os
import re
from typing import List, Tuple


ABS_PATH_RE = re.compile(r"([A-Za-z]:\\[^\"'\n]+|/Users/[^\"'\n]+|/home/[^\"'\n]+)")
STALE_HINTS = [
	"thesis-experiment",
	"experiment_v2",
	"C:/Users",
	"C:\\\\Users",
]
SCAN_EXTS = {".py", ".yml", ".yaml", ".md", ".txt", ".json", ".sh", ".bat"}


def parse_args() -> argparse.Namespace:
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	p = argparse.ArgumentParser(description="Audit hardcoded path references in repository scripts.")
	p.add_argument(
		"--root",
		default=repo_root,
		help="Repository root",
	)
	p.add_argument(
		"--targets",
		nargs="*",
		default=["analysis", "src", "config"],
		help="Top-level directories to scan",
	)
	p.add_argument(
		"--out",
		default=os.path.join(repo_root, "results", "validation", "path_reference_audit.md"),
		help="Output markdown report",
	)
	return p.parse_args()


def should_scan(path: str) -> bool:
	_, ext = os.path.splitext(path)
	return ext.lower() in SCAN_EXTS


def collect_findings(file_path: str) -> List[Tuple[int, str, str]]:
	findings: List[Tuple[int, str, str]] = []
	with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
		for i, line in enumerate(f, start=1):
			for m in ABS_PATH_RE.finditer(line):
				findings.append((i, "absolute-path", m.group(1).strip()))
			low = line.lower()
			for hint in STALE_HINTS:
				if hint.lower() in low:
					findings.append((i, "stale-hint", hint))
	return findings


def main() -> int:
	args = parse_args()
	repo_root = os.path.abspath(args.root)
	all_findings = []

	for target in args.targets:
		target_dir = os.path.join(repo_root, target)
		if not os.path.isdir(target_dir):
			continue
		for cur, _, files in os.walk(target_dir):
			for name in files:
				fp = os.path.join(cur, name)
				if not should_scan(fp):
					continue
				fs = collect_findings(fp)
				if fs:
					rel = os.path.relpath(fp, repo_root).replace("\\", "/")
					all_findings.append((rel, fs))

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("# Path Reference Audit\n\n")
		f.write(f"- Scanned root: `{repo_root.replace('\\\\', '/')}`\n")
		f.write(f"- Target dirs: {', '.join(args.targets)}\n")
		f.write(f"- Files with findings: {len(all_findings)}\n\n")

		if not all_findings:
			f.write("No hardcoded absolute paths or stale hints were detected.\n")
		else:
			for rel, findings in sorted(all_findings, key=lambda x: x[0]):
				f.write(f"## {rel}\n")
				for line_no, kind, token in findings:
					token_safe = token.replace("`", "'")
					f.write(f"- L{line_no} [{kind}] `{token_safe}`\n")
				f.write("\n")

	print(f"[OK] Audit report written: {args.out}")
	print(f"[INFO] Files with findings: {len(all_findings)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
