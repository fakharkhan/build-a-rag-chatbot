from langchain_text_splitters import MarkdownHeaderTextSplitter

# Chunk the document based on h2 headers.
with open("WonderVector5000.md", "r") as f:
    markdown_document = f.read()

headers_to_split_on = [
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

print(md_header_splits)
print("\n")
