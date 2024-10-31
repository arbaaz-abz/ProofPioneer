import json
import os
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import random
from utils.gemini_interface import GeminiAPI
# from utils.google_customsearch import GoogleCustomSearch
from utils.serper_customsearch import SerperCustomSearch


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


# def parse_llm_questions_response(response):
#     pattern = r'^[12]\.\s*\**(.+?)\**\s*$'
#     matches = re.findall(pattern, response.text, re.MULTILINE)
#     # Clean up any remaining markdown or whitespace
#     cleaned_questions = [question.strip() for question in matches]
#     return cleaned_questions

def extract_and_format_date(check_date, default_date="01-01-2022"):
    # If the date is not provided, use the default date
    if check_date != "UNKNOWN":
        date, month, year = check_date.split("-")
    else:
        date, month, year = default_date.split("-")
    
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
        # examples = [examples[2298]]
        examples = random.sample(examples, 10) # Pick 5 random examples for testing
        min_date = "01-01-2021" # DD-MM-YYYY

    # Question generation prompt
    with open("prompts/questions_template_v2.txt", "r") as f:
        question_prompt_template = f.read()
        response_schema_questions = {
            "type": "array",
            "items": {
                "type": "string",
            },
        }

    # Save folder
    save_folder = "outputs"
    create_output_folder(save_folder)

    # Google Search
    n_pages = 1
    max_google_search_calls_per_account = 100
    max_serper_calls_per_account = 2500
    # searcher = GoogleCustomSearch(max_api_calls_per_account, n_pages, "secrets/google_secrets.json")
    searcher = SerperCustomSearch(max_serper_calls_per_account, "secrets/serper_secrets.json")

    # Gemini Interface 
    gemini = GeminiAPI(model_name="gemini-1.5-pro-latest", 
                       temperature=0.5,
                       secrets_file="secrets/gemini_keys.json",
                       response_mime_type="application/json",
                       response_schema=response_schema_questions)

    results = {}
    claim_queries = {}

    # Create a CSV file to store the search results
    results_filename = f"{save_folder}/search_results.json"
    claim_queries_filename = f"{save_folder}/claim_queries.json"

    existing = {}

    for index, example in tqdm(enumerate(examples)):
        claim = example["claim"]
        if claim in existing:
            continue

        #claim = "Tony Hinchcliffe is getting backlash from his set at the MSG Trump Rally"
    
        print("PROCESSING CLAIM: ", claim)

        speaker = example["speaker"].strip() if "speaker" in example and example["speaker"] else "UNKNOWN"
        claim_date = example["claim_date"].strip() if "claim_date" in example and example["claim_date"] else "UNKNOWN"
        claim_types = ', '.join(example["claim_types"]) if "claim_types" in example and example["claim_types"] else ""
        location_ISO_code = example["location_ISO_code"].strip() if "location_ISO_code" in example and example["location_ISO_code"] else "US"

        # speaker = "The Hill"
        # claim_date = "2024-10-27"
        # claim_types = "Event/Property Claim" # Position Statement, Causal Claim, Numerical Claim
        # location_ISO_code = "US"

        print(speaker, claim_date, claim_types, location_ISO_code)

        # Generate questions using Gemini API
        prompt = question_prompt_template.replace("[Insert the claim here]", claim).replace("[Insert the claim speaker here]", speaker).replace("[Insert the claim date here]", claim_date).replace("[Insert the claim type here]", claim_types).replace("[Insert the location ISO code here]", location_ISO_code)
        response = gemini.get_llm_response(prompt)

        try:    
            llm_decompostions = json.loads(response.text)
            print("Generated ", len(llm_decompostions), " decompositions")
        except:
            print(response.text)
            continue

        # Extract and format the date
        sort_date = extract_and_format_date(claim_date, default_date=min_date)
        
        search_strings = []
        search_types = []

        # 1. Combine claim and author if available
        # if speaker is not None:
        #     search_string = string_to_search_query(f"{claim} {speaker}", None)
        #     search_strings.append(search_string)
        #     search_types.append("claim+author")

        # search_string = string_to_search_query(claim, None)
        # search_strings.append(search_string)
        # search_types.append("claim")

        # 3. Process and remove duplicate questions (if any)
        processed_queries = set()
        for decomp in llm_decompostions:
            # processed_question = string_to_search_query(question, None)
            # if processed_question not in search_strings:
            if decomp not in search_strings:
                processed_queries.add(decomp)
                search_strings.append(decomp)
                search_types.append("generated_decomposition")

        search_results = []
        visited = {}
        claim_queries[claim] = []
        
        store_counter = 0
        ts = []

        results[claim] = {}

        for this_search_string, this_search_type in zip(search_strings, search_types):
            # Bookkeeping
            claim_queries[claim].append((
                this_search_string,
                this_search_type
            ))

            sstring_search_results = searcher.fetch_results(this_search_string, "2021-01-01", location_ISO_code, n_pages)
            results[claim][this_search_string] = sstring_search_results

        # Save updated results and claim_queries after processing each claim
        with open(results_filename, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=4)

        with open(claim_queries_filename, "w", encoding="utf-8") as fp:
            json.dump(claim_queries, fp, indent=4)
