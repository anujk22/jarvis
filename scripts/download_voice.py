import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. jgkawell/jarvis")
    ap.add_argument("--filename", required=True, help="File in the repo, e.g. jarvis.onnx")
    ap.add_argument("--out", required=True, help="Output path to write the file to")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(repo_id=args.repo, filename=args.filename)
    out_path.write_bytes(Path(downloaded).read_bytes())
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

