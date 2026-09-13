import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
PUBLIC_DOCS = (
    "README.md",
    "RESEARCH_REPORT.md",
    "EXECUTIVE_SUMMARY.md",
    "PHASE5_AUDIT.md",
    "ROBUSTNESS_VERDICT.md",
    "RESUME_BULLETS.md",
    "ALPHAFORGE_INTERVIEW.md",
    "RECRUITER_AUDIT.md",
    "LEARN_THIS.md",
    "reports/README.md",
)


def test_phase7_documents_and_verdict_are_present():
    for relative_path in PUBLIC_DOCS:
        assert (ROOT / relative_path).is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "NO PERSISTENT SIGNAL" in readme
    assert "```mermaid" in readme
    assert "synthetic verification data" in readme


def test_resume_has_exactly_three_recommended_bullets():
    text = (ROOT / "RESUME_BULLETS.md").read_text(encoding="utf-8")
    recommended = text.split("# 30_SECOND_EXPLANATION", maxsplit=1)[0]
    assert sum(line.startswith("- ") for line in recommended.splitlines()) == 3


def test_public_document_links_resolve_and_paths_are_portable():
    link_pattern = re.compile(r"!?(?:\[[^]]*\])\(([^)]+)\)")
    drive_path_pattern = re.compile(r"[A-Za-z]:[\\/]")

    for relative_path in PUBLIC_DOCS:
        document = ROOT / relative_path
        text = document.read_text(encoding="utf-8")
        assert drive_path_pattern.search(text) is None
        for target in link_pattern.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (document.parent / target).resolve().exists(), (
                f"Broken link in {relative_path}: {target}"
            )


def test_public_claims_remain_scientifically_conservative():
    combined = "\n".join(
        (ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in PUBLIC_DOCS
    ).lower()
    for unsupported_claim in (
        "profitable strategy",
        "highly accurate",
        "predicts markets",
        "alpha discovered",
    ):
        assert unsupported_claim not in combined
