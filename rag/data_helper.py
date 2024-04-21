import pdftotext


class PDFReader:
    def __init__(self, pdf_paths: list[str] | str):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        self.pdf_path = pdf_paths

    def read(self) -> list[str]:
        texts = []
        for pdf_path in self.pdf_path:
            with open(pdf_path, "rb") as file:
                pdf = pdftotext.PDF(file)
                # Join all pages into a single string by \n\n
                texts.append("\n\n".join(pdf))
        return texts
