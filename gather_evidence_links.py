import json
import os
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import re
import random
from utils.gemini_interface import GeminiAPI
from utils.google_customsearch import GoogleCustomSearch


def string_to_search_query(text, author):
    parts = word_tokenize(text.strip())
    tags = pos_tag(parts)

    keep_tags = ["CD", "JJ", "NN", "VB"] # Cardinal Numbers, Adjectives, Nouns, Verbs

    if author is not None:
        search_string = author.split()
    else:
        search_string = []

    for token, tag in zip(parts, tags):
        for keep_tag in keep_tags:
            if tag[1].startswith(keep_tag):
                search_string.append(token)

    search_string = " ".join(search_string)
    return search_string


# Write a function to create an folder called outputs
def create_output_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def parse_llm_questions_response(response):
    pattern = r'^[12]\.\s*\**(.+?)\**\s*$'
    matches = re.findall(pattern, response.text, re.MULTILINE)
    # Clean up any remaining markdown or whitespace
    cleaned_questions = [question.strip() for question in matches]
    return cleaned_questions

def extract_and_format_date(check_date, default_date="2022-01-01"):

    # If the date is not provided, use the default date
    if check_date:
        year, month, date = check_date.split("-")
    else:
        year, month, date = default_date.split("-")
    
    if len(year) == 2 and int(year) <= 30:
        year = "20" + year
    elif len(year) == 2:
        year = "19" + year
    elif len(year) == 1:
        year = "200" + year

    month = month.zfill(2)
    date = date.zfill(2)

    return f"{year}{month}{date}"


if __name__ == "__main__":
    # Load a claims dataset
    with open("claim_datasets/averitec/train.json") as fp:
        examples = json.load(fp) 
        examples = random.sample(examples, 50) # Pick 50 random examples for testing
        min_date = "2022-01-01" # YYYY-MM-DD

    # Question generation prompt
    with open("prompts/prompt_2Q.txt", "r") as f:
        question_prompt_template = f.read()

    # Save folder
    save_folder = "outputs"
    create_output_folder(save_folder)

    # Google Search
    n_pages = 1
    max_api_calls_per_account = 100
    google_search = GoogleCustomSearch(max_api_calls_per_account, n_pages)

    # Gemini Flash 1.5 Interface 
    gemini_flash_api = GeminiAPI(model_name="gemini-1.5-flash-latest", secrets_file="secrets/gemini_keys.json")
    # gemini_pro_api = GeminiAPI(model_name="gemini-1.5-pro-latest")

    results = {}
    claim_queries = {}

    # Create a CSV file to store the search results
    results_filename = f"{save_folder}/search_results.json"
    claim_queries_filename = f"{save_folder}/claim_queries.json"
    # csv_header = ["index", "claim", "link", "page_num", "search_string", "search_type", "store_file_path"]

    existing = {}

    for index, example in tqdm(enumerate(examples)):
        claim = example["claim"]

        # If we already have this claim in the file we are resuming from, skip it
        if claim in existing:
            continue

        if "speaker" in example:
            speaker = example["speaker"].strip() if example["speaker"] else None
        else:
            speaker = None
        speaker = None # Remove speaker for now
        
        # Generate questions using Gemini API
        prompt = question_prompt_template.replace("[Insert the claim here]", claim)
        response = gemini_flash_api.get_llm_response(prompt)

        # Extract the questions from the response
        llm_questions = parse_llm_questions_response(response)

        # Extract and format the date
        sort_date = extract_and_format_date(example.get("claim_date", ""), default_date=min_date)
        
        search_strings = []
        search_types = []

        # 1. Combine claim and author if available
        if speaker is not None:
            search_string = string_to_search_query(f"{claim} {speaker}", None)
            search_strings.append(search_string)
            search_types.append("claim+author")

        search_string = string_to_search_query(claim, None)
        search_strings.append(search_string)
        search_types.append("claim")

        # 3. Process and remove duplicate questions (if any)
        processed_questions = set()
        for question in llm_questions:
            # processed_question = string_to_search_query(question, None)
            # if processed_question not in search_strings:
            if question not in search_strings:
                processed_questions.add(question)
                search_strings.append(question)
                search_types.append("generated_question")

        search_results = []
        visited = {}
        claim_queries[claim] = []
        
        store_counter = 0
        ts = []

        print("PROCESSING CLAIM: ", claim)
        results[claim] = {}

        for this_search_string, this_search_type in zip(search_strings, search_types):
            # Bookkeeping
            claim_queries[claim].append((
                this_search_string,
                this_search_type
            ))

            sstring_search_results = google_search.fetch_results(this_search_string, sort_date)
            results[claim][this_search_string] = sstring_search_results

        # Save updated results and claim_queries after processing each claim
        with open(results_filename, "w") as fp:
            json.dump(results, fp, indent=4)

        with open(claim_queries_filename, "w") as fp:
            json.dump(claim_queries, fp, indent=4)
