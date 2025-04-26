import requests
import xml.etree.ElementTree as ET
import time
import sys
import os

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUTPUT_DIR = "data/pubmed"

def get_pmc_full_text(query: str, max_results: int = 5, email: str | None = None) -> dict[str, str]:
    """
    Fetches full-text articles from PubMed Central (PMC) for a given query
    and saves them to the OUTPUT_DIR.

    Args:
        query: The search query (e.g., "covid vaccine effectiveness").
        max_results: The maximum number of articles to fetch.
        email: Your email address (recommended by NCBI for API usage).

    Returns:
        A dictionary where keys are PMCIDs and values are the extracted full text.
        Returns an empty dictionary if no results are found or an error occurs.
    """
    search_params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]", # Filter for PMC and open access
        "retmax": max_results,
        "usehistory": "y",
        "tool": "my_python_script", # Identify your script to NCBI
        "email": email or "default_user@example.com" # Provide an email
    }

    print(f"Searching PMC for: '{query}'...")
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 1. Search for PMCID/WebEnv
        search_response = requests.get(BASE_URL + "esearch.fcgi", params=search_params)
        search_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        time.sleep(0.4) # Respect NCBI rate limits (max 3/sec without API key)

        search_root = ET.fromstring(search_response.content)
        id_list = [id_elem.text for id_elem in search_root.findall(".//IdList/Id")]
        webenv = search_root.findtext(".//WebEnv")
        query_key = search_root.findtext(".//QueryKey")

        if not id_list or not webenv or not query_key:
            print("No articles found matching the query or required info missing.")
            return {}

        print(f"Found {len(id_list)} potential articles. Fetching full text...")

        # 2. Fetch full text using WebEnv/QueryKey
        fetch_params = {
            "db": "pmc",
            "query_key": query_key,
            "WebEnv": webenv,
            "rettype": "full", # Request full text format (often XML)
            "retmode": "xml",
            "retmax": max_results, # Redundant but safe
            "tool": "my_python_script",
            "email": email or "default_user@example.com"
        }

        fetch_response = requests.get(BASE_URL + "efetch.fcgi", params=fetch_params)
        fetch_response.raise_for_status()
        time.sleep(0.4)

        # 3. Parse full text XML and save
        full_texts = {}
        articles_saved_count = 0
        fetch_root = ET.fromstring(fetch_response.content)

        for article in fetch_root.findall('.//article'):
            pmcid_element = article.find('.//front/article-meta/article-id[@pub-id-type="pmc"]')
            if pmcid_element is None or pmcid_element.text is None:
                continue # Skip if PMCID is missing

            pmcid = f"PMC{pmcid_element.text}"

            # Extract text from body paragraphs (simple extraction)
            # More complex parsing might be needed depending on the desired content
            body_text = ""
            body_element = article.find('.//body')
            if body_element is not None:
                 # Join text content of all <p> tags within the body
                body_text = '\\n'.join(
                    ''.join(p.itertext()).strip() for p in body_element.findall('.//p') if p is not None
                )


            if body_text:
                full_texts[pmcid] = body_text
                # Save the text to a file
                filepath = os.path.join(OUTPUT_DIR, f"{pmcid}.txt")
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(body_text)
                    articles_saved_count += 1
                except IOError as e:
                    print(f"Error writing file {filepath}: {e}", file=sys.stderr)
            else:
                # Sometimes text might be in different structures
                # Add more specific extraction logic here if needed
                 print(f"Could not extract simple paragraph text for {pmcid}. XML structure might differ.")


        print(f"Successfully fetched text for {len(full_texts)} articles. Saved {articles_saved_count} to {OUTPUT_DIR}.")
        return full_texts

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during API request: {e}", file=sys.stderr)
        return {}
    except ET.ParseError as e:
        print(f"An error occurred during XML parsing: {e}", file=sys.stderr)
        # Log or inspect fetch_response.content here if needed
        # print(f"Failed XML content: {fetch_response.content.decode('utf-8', errors='ignore')[:500]}...")
        return {}
    except IOError as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return {}

if __name__ == "__main__":
    search_query = "psilocybin"
    # Optional: Replace with your actual email for NCBI
    user_email = "your.email@example.com"

    print(f"--- Running PubMed Central Full Text Fetcher ---")
    results = get_pmc_full_text(search_query, max_results=3, email=user_email)

    if results:
        print(f"--- Saved Articles in {OUTPUT_DIR} ---")
        for pmcid in results.keys():
             filepath = os.path.join(OUTPUT_DIR, f"{pmcid}.txt")
             # Check if file exists, in case saving failed for some reason
             if os.path.exists(filepath):
                 print(f"- {filepath}")
             else:
                 print(f"- {pmcid} (Text fetched but file not saved)")

    else:
        print("No full texts were retrieved or saved.")

    print("--- Fetcher Finished ---") 