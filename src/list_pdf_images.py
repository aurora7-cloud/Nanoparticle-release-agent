import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    path = Path(sys.argv[1])
    reader = PdfReader(path)
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except Exception as exc:
            print(page_index, "error", exc)
            continue
        print(page_index, len(images), [getattr(img, "name", "") for img in images[:5]])


if __name__ == "__main__":
    main()
