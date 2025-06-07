import os
import asyncio
import pandas as pd
from playwright.async_api import async_playwright

# Define the pages you want to scrape
urls = {
    "APL - aide au logement": "https://www.service-public.fr/particuliers/vosdroits/F12006",
    "Carte Vitale": "https://www.ameli.fr/assure/remboursements/tiers-payant/carte-vitale",
    "Demande RSA": "https://www.service-public.fr/particuliers/vosdroits/F1966",
    "Titre de séjour": "https://www.service-public.fr/particuliers/vosdroits/F16003"
}

async def scrape():
    data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for topic, url in urls.items():
            print(f"🔍 Scraping: {topic} → {url}")
            try:
                await page.goto(url, timeout=30000)
                await page.wait_for_selector("body", timeout=10000)

                # Extract all visible text from the body
                content = await page.inner_text("body")

                # Clean the text
                content = content.replace("\n", " ").replace("  ", " ").strip()

                # Optional: truncate long content
                if len(content) > 2000:
                    content = content[:2000]

                data.append({"topic": topic, "url": url, "content": content})

            except Exception as e:
                print(f"⚠️ Error on {url}: {e}")
                data.append({"topic": topic, "url": url, "content": "SCRAPE FAILED"})

        await browser.close()

    # Save results
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv("data/raw_scraped.csv", index=False, encoding="utf-8")
    print("✅ Saved to data/raw_scraped.csv")

if __name__ == "__main__":
    asyncio.run(scrape())
