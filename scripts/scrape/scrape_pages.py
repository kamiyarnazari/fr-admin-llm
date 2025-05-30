import requests
from bs4 import BeautifulSoup
import pandas as pd

# Add more pages as needed
urls = {
    "APL - aide au logement": "https://www.service-public.fr/particuliers/vosdroits/F12006",
    "Carte Vitale": "https://www.ameli.fr/assure/remboursements/tiers-payant/carte-vitale",
    "Demande RSA": "https://www.service-public.fr/particuliers/vosdroits/F1966",
    "Titre de séjour": "https://www.service-public.fr/particuliers/vosdroits/F16003"
}

data = []

for topic, url in urls.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50]
    content = "\n".join(paragraphs[:5])  # Limit to first 5 paragraphs

    data.append({
        "topic": topic,
        "url": url,
        "content": content
    })

df = pd.DataFrame(data)
df.to_csv("data/raw_scraped.csv", index=False, encoding="utf-8")
print(df.head())
