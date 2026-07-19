import asyncio
from playwright.async_api import async_playwright
import os, sys

async def convert(html_file, pdf_file):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        html_path = f"file:///{os.path.abspath(html_file)}".replace('\\', '/')
        await page.goto(html_path, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        await page.pdf(path=pdf_file, format="A4")
        await browser.close()

if __name__ == '__main__':
    # Default: convert eda_summary. Pass args for other files.
    if len(sys.argv) >= 3:
        html_in = sys.argv[1]
        pdf_out = sys.argv[2]
    else:
        html_in = "reports/eda_summary.html"
        pdf_out = "reports/eda_summary.pdf"
    asyncio.run(convert(html_in, pdf_out))
