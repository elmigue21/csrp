from code_preview import STEPS, _source, code_preview


def test_every_step_shows_real_code():
    for _, file, name, *_ in STEPS:
        code, start = _source(file, name)
        assert name in code and start > 0


def test_page_lists_every_step_without_results(tmp_path):
    page = code_preview(tmp_path)                    # no results.csv there: no results box
    assert page.count("<section") == len(STEPS) and "Result these steps produce" not in page
