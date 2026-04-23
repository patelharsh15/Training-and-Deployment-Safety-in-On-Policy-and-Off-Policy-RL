"""Use Playwright headless Chromium to print HTML → PDF on a headless Linux server."""
import asyncio
from playwright.async_api import async_playwright
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "Final_Project_Report.html")
PDF_PATH = os.path.join(BASE_DIR, "Final_Project_Report.pdf")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(f"file://{HTML_PATH}", wait_until="networkidle")
        await page.pdf(
            path=PDF_PATH,
            format="Letter",
            margin={"top": "0.75in", "bottom": "0.75in", "left": "0.75in", "right": "0.75in"},
            print_background=True,
            display_header_footer=True,
            header_template='<span></span>',
            footer_template='<div style="font-size:9px;width:100%;text-align:center;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
        )
        await browser.close()
    sz = os.path.getsize(PDF_PATH) / 1024 / 1024
    print(f"PDF generated: {PDF_PATH}")
    print(f"File size: {sz:.1f} MB")

asyncio.run(main())
