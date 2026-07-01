"""Seed DocuTrust with a sample client + policy document.

Generates a small, internally-consistent policy PDF at
``data/sample_policy.pdf`` and ingests it via the API. Run with:

    python -m scripts.seed_corpus
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx

log = logging.getLogger("docutrust.seed")
logging.basicConfig(level=logging.INFO, format="%(message)s")

API_BASE = os.environ.get("DOCUTRUST_API", "http://127.0.0.1:8000")
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_policy.pdf"


SAMPLE_POLICY = """
ACME Corp — Information Security Policy v3.2

1. Purpose and Scope
This Information Security Policy applies to all employees, contractors, and
third parties of ACME Corp who access corporate information systems, customer
data, or proprietary intellectual property. The policy establishes minimum
security controls to protect confidentiality, integrity, and availability of
data.

2. Access Control
2.1 All access to production systems requires multi-factor authentication
    (MFA). SMS-based MFA is permitted only as a fallback when hardware tokens
    or authenticator apps are unavailable, and must be rotated quarterly.
2.2 Privileged accounts (e.g. database administrators, production deployers)
    must use a dedicated privileged access workstation (PAW) and must rotate
    credentials every 90 days. Shared service accounts are prohibited.
2.3 Access reviews are conducted quarterly. Managers must attest to the
    continued need for each user's access within 10 business days of the
    review request.

3. Data Classification
3.1 Data is classified into four tiers: Public, Internal, Confidential, and
    Restricted. Customer personal data is, at minimum, Confidential. Payment
    card data is always Restricted.
3.2 Restricted data must be encrypted at rest using AES-256 or stronger, and
    in transit using TLS 1.2 or stronger. Key material must be stored in a
    FIPS 140-2 Level 2 (or higher) validated key management system.

4. Incident Response
4.1 Suspected security incidents must be reported to the Security Operations
    Center within 1 hour of detection via the on-call hotline or the
    internal #sec-incident Slack channel.
4.2 The Security Operations Center will triage, contain, and remediate the
    incident following NIST SP 800-61 Rev. 2. A post-incident review must be
    completed within 14 calendar days of resolution.

5. Vendor Risk Management
5.1 All vendors that process Confidential or Restricted data must complete a
    vendor security assessment prior to contract execution, and annually
    thereafter. Assessments are owned by the business sponsor.
5.2 Vendors are required to notify ACME within 72 hours of any confirmed data
    breach affecting ACME data, in line with applicable regulations.

6. Training
6.1 All employees must complete annual security awareness training. New hires
    must complete training within 30 days of start date.
6.2 Engineering staff must additionally complete secure development training
    covering OWASP Top 10, secrets management, and dependency hygiene.
"""


def _write_sample_pdf() -> Path:
    """Create a minimal PDF for the seed. We use reportlab if available,
    else fall back to a plain-text PDF built from the raw bytes (pypdf
    parses it). The plain-text fallback keeps the seed self-contained."""
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(SAMPLE_PATH), pagesize=LETTER)
        width, height = LETTER
        y = height - 72
        for paragraph in SAMPLE_POLICY.strip().split("\n\n"):
            for line in paragraph.split("\n"):
                if y < 72:
                    c.showPage()
                    y = height - 72
                c.drawString(72, y, line[:100])
                y -= 14
            y -= 8
        c.save()
        log.info("Wrote %s", SAMPLE_PATH)
        return SAMPLE_PATH
    except ImportError:
        pass

    # Fallback: write a tiny one-page PDF with the policy text as a single
    # content stream. pypdf extracts it fine; the grader is robust to plain
    # text.
    text = SAMPLE_POLICY.strip().replace("(", "[").replace(")", "]")
    stream = f"BT /F1 11 Tf 72 720 Td ({text}) Tj ET"
    content = stream
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Count 1 /Kids [3 0 R]>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
        b"4 0 obj <</Length " + str(len(content)).encode() + b">> stream\n"
        + content.encode("latin-1", errors="replace") + b"\nendstream endobj\n"
        b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000054 00000 n \n"
        b"0000000101 00000 n \n"
        b"0000000200 00000 n \n"
        b"0000000290 00000 n \n"
        b"trailer <</Size 6 /Root 1 0 R>>\n"
        b"startxref\n350\n%%EOF\n"
    )
    SAMPLE_PATH.write_bytes(body)
    log.info("Wrote minimal PDF fallback to %s", SAMPLE_PATH)
    return SAMPLE_PATH


async def _wait_for_api(client: httpx.AsyncClient) -> None:
    for _ in range(50):
        try:
            r = await client.get(f"{API_BASE}/healthz", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"API at {API_BASE} did not become ready in time")


async def main() -> int:
    SAMPLE_PATH = _write_sample_pdf()

    async with httpx.AsyncClient(timeout=60.0) as client:
        await _wait_for_api(client)

        # Create or reuse the demo client.
        r = await client.get(f"{API_BASE}/clients")
        r.raise_for_status()
        existing = [c for c in r.json() if c["name"] == "ACME Corp (Demo)"]
        if existing:
            client_id = existing[0]["_id"]
            log.info("Reusing demo client %s", client_id)
        else:
            r = await client.post(
                f"{API_BASE}/clients",
                json={"name": "ACME Corp (Demo)", "industry": "Manufacturing"},
            )
            r.raise_for_status()
            client_id = r.json()["_id"]
            log.info("Created demo client %s", client_id)

        with SAMPLE_PATH.open("rb") as fh:
            r = await client.post(
                f"{API_BASE}/ingest",
                data={"client_id": client_id},
                files={"file": (SAMPLE_PATH.name, fh, "application/pdf")},
            )
        if r.status_code != 200:
            log.error("Ingest failed: %s", r.text)
            return 1
        result = r.json()
        log.info(
            "Ingested %s: %d sections, %d chunks",
            result["filename"],
            result["section_count"],
            result["chunk_count"],
        )
        log.info("Done. Try: curl -X POST %s/ask -H 'Content-Type: application/json' \\", API_BASE)
        log.info("     -d '{\"client_id\": \"%s\", \"question\": \"How long do I have to report a security incident?\"}'", client_id)
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
