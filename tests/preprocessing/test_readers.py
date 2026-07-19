from pathlib import Path

from pypdf import PdfWriter

from project_genesis.datasets import DatasetSource, DatasetSplit, SourceFormat
from project_genesis.preprocessing import RawDocument, ReadFailure, read_source


def source(path: Path, format: SourceFormat, **options: object) -> DatasetSource:
    return DatasetSource(
        path.resolve(),
        format,
        DatasetSplit.TRAIN,
        **options,  # type: ignore[arg-type]
    )


def documents(results: list[RawDocument | ReadFailure]) -> list[RawDocument]:
    assert not any(isinstance(result, ReadFailure) for result in results)
    return [result for result in results if isinstance(result, RawDocument)]


def test_text_markdown_and_git_sources_are_read_deterministically(tmp_path: Path) -> None:
    text = tmp_path / "plain.txt"
    text.write_text("plain", encoding="utf-8")
    markdown = tmp_path / "notes.md"
    markdown.write_text("# Notes", encoding="utf-8")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "b.py").write_text("print('b')", encoding="utf-8")
    (repository / "a.py").write_text("print('a')", encoding="utf-8")
    (repository / "ignored.bin").write_bytes(b"\x00")

    assert (
        documents(list(read_source(source(text, SourceFormat.TEXT), root=tmp_path)))[0].text
        == "plain"
    )
    assert (
        documents(list(read_source(source(markdown, SourceFormat.MARKDOWN), root=tmp_path)))[0].text
        == "# Notes"
    )
    git_documents = documents(
        list(
            read_source(
                source(repository, SourceFormat.GIT, include_extensions=(".py",)),
                root=tmp_path,
            )
        )
    )
    assert [document.source for document in git_documents] == ["repo/a.py", "repo/b.py"]


def test_structured_and_web_sources_preserve_text_and_metadata(tmp_path: Path) -> None:
    json_file = tmp_path / "data.json"
    json_file.write_text(
        '[{"body": "first", "score": 1}, {"body": "second", "tags": ["a", "b"]}]',
        encoding="utf-8",
    )
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"text": "line one"}\n{"text": "line two"}\n', encoding="utf-8")
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("content,language\ncsv text,en\n", encoding="utf-8")
    html = tmp_path / "page.html"
    html.write_text(
        "<html><style>hidden</style><body>Hello <b>web</b></body></html>",
        encoding="utf-8",
    )

    json_documents = documents(
        list(
            read_source(
                source(json_file, SourceFormat.JSON, text_field="body"),
                root=tmp_path,
            )
        )
    )
    assert [document.text for document in json_documents] == ["first", "second"]
    assert json_documents[0].metadata["score"] == 1
    assert [
        document.text
        for document in documents(
            list(read_source(source(jsonl, SourceFormat.JSONL), root=tmp_path))
        )
    ] == ["line one", "line two"]
    assert (
        documents(
            list(
                read_source(
                    source(csv_file, SourceFormat.CSV, text_field="content"),
                    root=tmp_path,
                )
            )
        )[0].metadata["language"]
        == "en"
    )
    assert (
        documents(list(read_source(source(html, SourceFormat.WEB), root=tmp_path)))[0].text
        == "Hello  web"
    )


def test_pdf_source_yields_one_document_per_page(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf.open("wb") as stream:
        writer.write(stream)

    parsed = documents(list(read_source(source(pdf, SourceFormat.PDF), root=tmp_path)))

    assert len(parsed) == 1
    assert parsed[0].document_id == "document.pdf#page-1"
    assert parsed[0].metadata["page"] == 1


def test_malformed_source_returns_a_failure(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")

    results = list(read_source(source(malformed, SourceFormat.JSON), root=tmp_path))

    assert len(results) == 1
    assert isinstance(results[0], ReadFailure)
