from llama_index.core import Document
from pathlib import Path


def load_line_by_line_with_metadata(file_path_1: Path | str, file_path_2: Path | str) -> list[Document]:
    documents = []
    with open(file_path_1) as f1, open(file_path_2) as f2:
        for line1, line2 in zip(f1, f2):
            metadata = {"comment": line2.rstrip("\n")}
            documents.append(Document(text=line1.rstrip("\n"), metadata=metadata))
    return documents

