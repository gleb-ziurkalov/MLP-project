from concurrent.futures import ThreadPoolExecutor
import os
import time
from urllib.parse import urlparse, unquote
from duckduckgo_search import DDGS
import requests

def fetch_search_results(base_query, total_results=1000, batch_size=50, delay=2):
    """
    Fetch PDF links from DuckDuckGo using query variations and regional settings.
    """
    pdf_links = set()
    regions = ["wt-wt", "us-en", "de-de", "fr-fr", "cn-zh"]  # Regions to query
    keywords = [
        "RoHS substance compliance",
        "RoHS EU substance compliance",
        "RoHS China substance compliance",
        "TSCA substance compliance",
        "REACH substance compliance",
        "PROP65 substance compliance",
        "Proposition 65 substance compliance",
        "RoHS compliance statement",
        "RoHS EU compliance statement",
        "RoHS China compliance statement",
        "TSCA compliance statement",
        "REACH compliance statement",
        "PROP65 compliance statement",
        "Proposition 65 compliance statement",
    ]

    with DDGS() as ddgs:
        for region in regions:
            for keyword in keywords:
                query = f"{keyword} statement filetype:pdf"
                try:
                    results = ddgs.text(query, safesearch="Moderate", max_results=batch_size, region=region)
                    for result in results:
                        if result.get("href", "").endswith(".pdf"):
                            pdf_links.add(result["href"])
                    
                    print(f"Fetched {len(pdf_links)}/{total_results} unique PDF links so far...")
                    if len(pdf_links) >= total_results:
                        return sorted(list(pdf_links))[:total_results]
                    
                    time.sleep(delay)  # Respect rate limits
                except Exception as e:
                    print(f"Error fetching results for query '{query}' in region '{region}': {e}")
                    time.sleep(5)  # Retry delay on error

    return sorted(list(pdf_links))[:total_results]


def download_pdf(url, save_dir='pdfs'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    filename = unquote(filename)
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    file_path = os.path.join(save_dir, filename)
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def download_pdfs_in_parallel(pdf_links, save_dir='pdfs', max_threads=10):
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(download_pdf, url, save_dir) for url in pdf_links]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading file: {e}")

def main():
    search_term = "substance compliance statement filetype:pdf"
    total_pdfs = 1000  # Target number of PDFs
    pdf_links = fetch_search_results(search_term, total_results=total_pdfs, batch_size=50, delay=2)
    print(f"Total unique PDF links fetched: {len(pdf_links)}")
    download_pdfs_in_parallel(pdf_links, max_threads=20)

if __name__ == "__main__":
    main()
