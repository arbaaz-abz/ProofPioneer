import json
import os
import gc
import sqlite3
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
import multiprocessing

# Import the necessary function from your utils
from utils.webpage_crawler_v2 import url2lines

# Blacklists
BLACKLIST_DOMAINS = {
    "jstor.org",
    "facebook.com",
    "ftp.cs.princeton.edu",
    "nlp.cs.princeton.edu",
    "huggingface.co",
}

BLACKLIST_FILES = [
    "/glove.",
    "ftp://ftp.cs.princeton.edu/pub/cs226/autocomplete/words-333333.txt",
    "https://web.mit.edu/adamrose/Public/googlelist",
]

def create_directory_chain(path):
    os.makedirs(path, exist_ok=True)

def get_domain_name(url):
    if '://' not in url:
        url = 'http://' + url
    domain = urlparse(url).netloc
    return domain[4:] if domain.startswith("www.") else domain

def should_filter_link(link):
    domain = get_domain_name(link)
    if domain in BLACKLIST_DOMAINS:
        return True
    if any(b_file in link for b_file in BLACKLIST_FILES):
        return True
    if link.endswith((".pdf", ".doc")):
        return True
    return False

def initialize_db(db_path="index.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_table (
            key TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            original_url TEXT,
            crawl_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def insert_index(conn, key, path, original_url, lock):
    try:
        with lock:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO index_table (key, path, original_url) VALUES (?, ?, ?)", 
                (key, path, original_url)
            )
            conn.commit()
    except sqlite3.IntegrityError:
        print(f"Duplicate entry for key: {key}")
    except Exception as e:
        print(f"Failed to insert index for {original_url}: {e}")

def get_and_store(link, store_file_path):
    try:
        page_lines = url2lines(link)
        content = "\n".join([link] + page_lines) if page_lines else "NA"
        with open(store_file_path, "w", encoding='utf-8') as out_f:
            out_f.write(content)
        gc.collect()
        print(f"Successfully processed: {link}")
        return True, link, store_file_path
    except Exception as e:
        print(f"Error processing {link}: {e}")
        return False, link, store_file_path

def worker_task(args):
    link, store_file_path = args
    return get_and_store(link, store_file_path)

def main():
    # Load search results
    with open("outputs/search_results.json", "r") as fp:
        search_results = json.load(fp)

    # Setup storage
    store_folder = "outputs/webpages/"
    create_directory_chain(store_folder)

    # Initialize database
    conn = initialize_db("outputs/index.db")

    # Initialize multiprocessing manager and shared data structures
    manager = Manager()
    visited = manager.dict()
    lock = Lock()

    # Initialize variables
    store_counter = 0
    arguments = set()          # To store unique (link, path) tuples for processing
    index_entries = []     # To store all (key, path) tuples for indexing

    # Prepare tasks
    for claim_index, (claim, queries) in enumerate(tqdm(search_results.items())):
        for query_index, (page_num, page_results) in enumerate(queries.items()):
            for page_num, results in page_results.items():
                for webpage_index, result_object in enumerate(results):
                    link = str(result_object["link"]).strip()
                    if should_filter_link(link):
                        continue

                    key = f"{claim_index}_{query_index}_{page_num}_{webpage_index}"

                    if link in visited:
                        store_file_path = visited[link]
                    else:
                        store_counter += 1
                        store_file_path = os.path.join(
                            store_folder, f"webpage_{store_counter}.txt"
                        )
                        visited[link] = store_file_path
                        arguments.add((link, store_file_path))
                    
                    index_entries.append((key, store_file_path, link))
        # if claim_index == 1:
        #     break

    print(f"Total unique links to process: {len(arguments)}")
    print(f"Total key-path mappings to insert: {len(index_entries)}")

    # Define the number of processes based on CPU count
    cpu_count = multiprocessing.cpu_count() - 1
    print(f"Starting multiprocessing with {cpu_count} processes.")

    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # Submit all unique link processing tasks
        future_to_args = {executor.submit(worker_task, arg): arg for arg in arguments}
        
        # Iterate over completed futures as they finish
        for future in tqdm(as_completed(future_to_args), total=len(future_to_args)):
            success, url, path = future.result()
            if not success:
                print(f"Failed to process link: {url}")

    # Insert all key-path mappings into the database
    print("Starting database insertion of key-path mappings.")
    for key, path, url in tqdm(index_entries, desc="Inserting into Database"):
        insert_index(conn, key, path, url, lock)

    conn.close()
    print("Processing complete.")

if __name__ == "__main__":
    main()