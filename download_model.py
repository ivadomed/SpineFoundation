#!/usr/bin/env python3
"""
Download a HuggingFace model to the local models/ directory.

Token setup
-----------
1. Create a HuggingFace account at https://huggingface.co
2. Generate a READ token at https://huggingface.co/settings/tokens
3. Save it to  models/token.txt  (one line, no spaces):

       models/
       └── token.txt       ← hf_xxxxxxxxxxxxxxxxxxxxxxxx
           curia/          ← downloaded model (auto-created)

   The models/ folder is in .gitignore — your token will never be pushed.

Usage
-----
    python download_model.py --model raidium/curia
    python download_model.py --model raidium/curia --output_dir ./models/curia
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


TOKEN_FILE = Path(__file__).parent / "models" / "token.txt"


def load_token() -> str:
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"Token file not found: {TOKEN_FILE}\n"
            "Create it and paste your HuggingFace READ token inside.\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"{TOKEN_FILE} is empty — paste your HuggingFace token inside.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model using the token stored in models/token.txt"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace repo id, e.g. raidium/curia",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Local destination folder. Defaults to models/<model_name>/",
    )
    args = parser.parse_args()

    token = load_token()
    print(f"Token loaded from {TOKEN_FILE}")

    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else Path("models") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading  : {args.model}")
    print(f"Destination  : {output_dir.resolve()}")

    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_dir),
        token=token,
    )

    print(f"\nModel saved to {output_dir.resolve()}")
    print(f"Use it in analyze_embeddings.py with:  --model {output_dir}")


if __name__ == "__main__":
    main()
