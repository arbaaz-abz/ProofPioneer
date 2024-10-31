import requests
import json
from itertools import cycle

class SerperCustomSearch:
    def __init__(self, max_calls, secrets_file):
        self.max_calls = max_calls
        self.serper_api_counter = 0
        self.serper_url = "https://google.serper.dev/search"
        self.keys_pool = self._load_secrets(secrets_file)
        self.api_key = None

    def _load_secrets(self, secrets_file):
        with open(secrets_file) as fp:
            serper_api_secrets = json.load(fp)
        return cycle(serper_api_secrets)
    
    def _rotate_secret(self):
        current_secret = next(self.keys_pool)
        self.api_key = current_secret["api_key"]
        self.serper_api_counter = 0
        print("Changing Serper search SECRET")

    def _get_serper_search_results(self, search_string, location_ISO_code, page):
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = json.dumps({
            "q": search_string,
            "gl": location_ISO_code.lower(),
            "page": page
        })
        print(payload, self.api_key)

        try:
            response = requests.post(self.serper_url, headers=headers, data=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return {"organic": []}
        
    def fetch_results(self, search_string, pages_before_date, location_ISO_code, n_pages):
        search_results = {}
        search_string = f"{search_string} before:{pages_before_date}"
        for page_num in range(n_pages):
            if self.serper_api_counter > self.max_calls or not self.api_key:
                self._rotate_secret()

            page_results = self._get_serper_search_results(search_string, location_ISO_code, page_num+1)
            search_results[page_num+1] = page_results["organic"]
            self.serper_api_counter += 1
        return search_results

if __name__ == "__main__":
    scs = SerperCustomSearch(2500, "../secrets/serper_secrets.json")
    # min_date = "2021-01-01"
    min_date = "2021-01-01"
    results = scs.fetch_results("Unemployment rate in South Bend, Indiana",
                                min_date,
                                "US",
                                1)
    
    with open("serper_test.json", "w") as fp:
        json.dump(results, fp, indent=4)