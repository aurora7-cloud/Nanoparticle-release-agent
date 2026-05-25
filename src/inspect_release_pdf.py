import re
import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    path = Path(sys.argv[1])
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"pages {len(reader.pages)}")
    print(f"chars {len(text)}")
    for pattern in [
        "release",
        "Release",
        "pH",
        "DTXL",
        "DOXY",
        "encapsulation",
        "particle size",
        "zeta",
        "Fig. 1",
    ]:
        print(f"\n--- {pattern}")
        for match in list(re.finditer(re.escape(pattern), text))[:8]:
            start = max(0, match.start() - 350)
            end = min(len(text), match.end() + 500)
            snippet = text[start:end].replace("\n", " ")
            print(snippet[:1200])


if __name__ == "__main__":
    main()
