import os
import pandas as pd
import trafilatura

# List of target French administrative URLs
urls = {
    "Carte Vitale": "https://forum-assures.ameli.fr/questions/3571681-appli-carte-vitale"
}

data = []

for topic, url in urls.items():
    print(f"Scraping: {topic} → {url}")
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        result = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if result:
            cleaned = result.strip().replace("\n", " ").replace("  ", " ")
            if len(cleaned) > 200:
                data.append({"topic": topic, "url": url, "content": cleaned[:2000]})
            else:
                data.append({"topic": topic, "url": url, "content": "TOO SHORT"})
        else:
            data.append({"topic": topic, "url": url, "content": "EXTRACTION FAILED"})
    else:
        data.append({"topic": topic, "url": url, "content": "FETCH FAILED"})

# Save to data/raw_scraped.csv
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(data)
df.to_csv("data/raw_scraped.csv", index=False, encoding="utf-8")
print("✅ Saved to data/raw_scraped.csv")
