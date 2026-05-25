import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(path)
    for page_index, page in enumerate(reader.pages, start=1):
        for image_index, image in enumerate(page.images, start=1):
            suffix = Path(image.name).suffix or ".jpg"
            out = out_dir / f"page_{page_index:02d}_image_{image_index:02d}{suffix}"
            out.write_bytes(image.data)
            print(out)


if __name__ == "__main__":
    main()
