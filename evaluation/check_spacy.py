"""Run on a machine with spaCy + en_core_web_sm installed (e.g. your Windows PC):  python -m evaluation.check_spacy

Reports which NLP backend is active and compares the two relation modes on the DEV lectures only:
rules (default, verified) vs parse (opt-in dependency-parse extractor, only stub-tested in the dev sandbox)."""
import os

from lecturenotes import nlp

from . import run_eval as E


def main():
    print("backend:", nlp.backend())
    if nlp.backend() != "spacy":
        print("spaCy model not loaded: run  pip install spacy  and  python -m spacy download en_core_web_sm")
        return
    for mode in ("rules", "parse"):
        os.environ["LN_RELATIONS"] = mode
        rows = []
        for gold, text in E.load("dev"):
            for vname, vtext in E.variants(gold, text):
                rows.append(E.evaluate_one(gold, text, vname, vtext)[0])
        print(f"\n=== relation mode: {mode} ===")
        print(E.table(rows)[0])


if __name__ == "__main__":
    main()
