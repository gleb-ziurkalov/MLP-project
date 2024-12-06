import requests
import os
from urllib.parse import urlparse, unquote, urlencode
import time

# Google Custom Search API credentials
API_KEY = 'AIzaSyBrxxDL6WFrO8LxaP754h0Nbp5BjANsAUY'
CSE_ID = 'f0110ae278e434591'

# Search settings
NUM_RESULTS = 500  # Total number of results you want to fetch (up to 100)
RESULTS_PER_REQUEST = 10  # Google Custom Search API allows a max of 10 results per request

def google_search(search_term, api_key, cse_id, start_index, num=10, print_url=False):
    """
    Performs a Google Custom Search API request.
    """
    service_url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': search_term,
        'cx': cse_id,
        'key': api_key,
        'num': num,
        'start': start_index,  # Start index for search results
        'fileType': 'pdf',
        'exactTerms': 'compliance statement',
        'gl': 'de',  # Geolocation for Germany
    }
    if print_url:
        # Print the constructed URL for debugging purposes
        query_string = urlencode(params)
        full_url = f"{service_url}?{query_string}"
        print(f"API Call: {full_url}")

    # Make the request
    response = requests.get(service_url, params=params)
    response.raise_for_status()
    return response.json()

def download_pdf(url, save_dir='pdfs'):
    """
    Downloads a PDF file from the given URL and saves it to the specified directory.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Parse the URL to generate a clean filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)  # Extract the base filename from the URL path
    filename = unquote(filename)  # Decode URL encoded characters

    # Remove invalid characters for filenames on the filesystem
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')

    # Create the path where the file will be saved
    local_filename = os.path.join(save_dir, filename)

    # Download the PDF
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'Downloaded: {local_filename}')
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def main():
    query = 'substance compliance statement pdf'  # The search query
    total_results = 0
    start_index = 1  # Start from the first result

    while total_results < NUM_RESULTS:
        try:
            # Adding print_url=True will print the constructed API URL
            results = google_search(query, API_KEY, CSE_ID, start_index, RESULTS_PER_REQUEST, print_url=True)
            if 'items' not in results:
                print("No more results found.")
                break

            # Iterate over search results and download each PDF link found
            for item in results.get('items', []):
                pdf_url = item.get('link')
                try:
                    print(f"Downloading PDF from URL: {pdf_url}")
                    download_pdf(pdf_url)  # Download the PDF
                except Exception as e:
                    print(f'Failed to download {pdf_url}: {e}')

            # Update the total results fetched and increment the start index for the next batch
            total_results += len(results.get('items', []))
            start_index += RESULTS_PER_REQUEST

            # Delay to avoid hitting Google API rate limits
            time.sleep(5)

        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Detailed HTTP error message
            break
        except Exception as err:
            print(f'Other error occurred: {err}')  # General error handling
            break

if __name__ == '__main__':
    main()
