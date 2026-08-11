from app.tools.arxiv import parse_arxiv_feed


def test_parse_arxiv_feed_normalizes_metadata() -> None:
    payload = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
        <title>  A   Research Paper  </title>
        <summary> An abstract with   whitespace. </summary>
        <published>2024-01-15T00:00:00Z</published>
        <updated>2024-02-01T00:00:00Z</updated>
        <author><name>Jane Doe</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2401.12345" />
      </entry>
    </feed>"""

    papers = parse_arxiv_feed(payload)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.12345v2"
    assert papers[0].title == "A Research Paper"
    assert papers[0].abstract == "An abstract with whitespace."
    assert papers[0].authors == ["Jane Doe"]
