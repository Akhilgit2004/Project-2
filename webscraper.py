import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from tqdm import tqdm

def scrape_1800s_tamil_authors():
    # 1. THE TARGET LIST (Tamil Renaissance & 19th Century Authors)
    # The script will look for these exact names in the Project Madurai index
    target_authors = [
        "வேதநாயகம் பிள்ளை",       # Mayuram Vedanayagam Pillai (1826–1889)
        "ஆறுமுக நாவலர்",         # Arumuka Navalar (1822–1879)
        "சுவாமிநாத ஐயர்",        # U. V. Swaminatha Iyer (1855–1942)
        "ராஜம் ஐயர்",            # B. R. Rajam Iyer (1872–1898)
        "மாதவையா",              # A. Madhaviah (1872–1925)
        "மறைமலை அடிகள்",         # Maraimalai Adigal (1876–1950)
        "சுந்தரம் பிள்ளை",          # P. Sundaram Pillai (1855–1897)
        "பரிதிமாற் கலைஞர்",       # Parithimar Kalaignar / V.G.S Sastri (1870–1903)
        "சுப்பிரமணிய பாரதி",      # Subramania Bharati (1882–1921)
        "பாரதியார்",             # Bharatiyar (Alt spelling)
        "வ. உ. சிதம்பரம்பிள்ளை"  # V.O. Chidambaram Pillai (1872–1936)
    ]
    
    print("🔍 Step 1: Scanning Project Madurai Master Index for 19th Century Authors...")
    index_url = "https://www.projectmadurai.org/pmworks.html"
    response = requests.get(index_url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 2. MATCH AUTHORS TO DOCUMENT URLs
    # The index is a giant HTML table. We need to find the author column and the URL column.
    target_urls = []
    rows = soup.find_all('tr')
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 6: # Ensure it's a valid data row
            author_text = cols[2].get_text().strip()
            
            # Check if any of our target authors are in this row
            if any(target in author_text for target in target_authors):
                # Find the Unicode HTML link (usually in the 6th column)
                links = cols[5].find_all('a')
                for link in links:
                    href = link.get('href')
                    if href and href.endswith('.html') and 'pmuni' in href:
                        full_url = f"https://www.projectmadurai.org/{href.replace('./', '')}"
                        target_urls.append({"author": author_text, "url": full_url})

    print(f"✅ Found {len(target_urls)} historical texts matching your authors!")
    
    # 3. SCRAPE ONLY THE TARGETED TEXTS
    results = []
    print(f"\n🚀 Step 2: Extracting raw Tamil text from {len(target_urls)} targeted links...")
    
    for doc in tqdm(target_urls):
        try:
            doc_resp = requests.get(doc['url'], timeout=10)
            doc_resp.encoding = 'utf-8'
            doc_soup = BeautifulSoup(doc_resp.text, 'html.parser')
            
            # Extract text blocks
            text_blocks = doc_soup.get_text(separator='\n\n').split('\n\n')
            
            for block in text_blocks:
                cleaned_block = block.strip()
                
                # FILTER: Must be > 80 chars, contain Tamil script, and avoid English headers
                if len(cleaned_block) > 80 and re.search(r'[\u0B80-\u0BFF]', cleaned_block):
                    cleaned_block = re.sub(r'\s+', ' ', cleaned_block)
                    results.append({
                        "author": doc['author'],
                        "source_url": doc['url'],
                        "text": cleaned_block
                    })
                    
        except Exception as e:
            print(f"⚠️ Error on {doc['url']}: {e}")
            
        time.sleep(1) # Be polite to the server
        
    # 4. SAVE TO CSV
    df = pd.DataFrame(results)
    
    # Drop duplicates just in case the same paragraph got parsed twice
    df = df.drop_duplicates(subset=['text']) 
    
    output_filename = "madurai_1800s_tamil_raw.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n🎉 SUCCESS! Scraped {len(df)} pure Tamil paragraphs from the 1800s.")
    print(f"💾 Saved to {output_filename}")

if __name__ == "__main__":
    scrape_1800s_tamil_authors()