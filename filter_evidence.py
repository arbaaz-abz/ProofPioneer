import json
import sqlite3
import nltk
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from utils.gemini_interface import GeminiAPI

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

def load_unwanted_patterns(config_path: str) -> List[re.Pattern]:
    with open(config_path, "r") as f:
        patterns = json.load(f)
    return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]


def retrieve_all_rows(db_path: str = "outputs/index.db") -> List[Dict[str, Any]]:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()            
            cursor.execute("SELECT * FROM index_table")
            rows = cursor.fetchall()                        
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Error accessing the database: {e}")
        return []

def retrieve_row_by_key(key: str, db_path: str = "outputs/index.db") -> Dict[str, Any]:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM index_table WHERE key = ?", (key,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"Error accessing the database: {e}")
        return None

def extract_relevant_sentences(sentences: List[str], model: SentenceTransformer, claim_embedding: Any, threshold: float = 0.65) -> (List[int], List[float]):
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity_scores = util.cos_sim(claim_embedding, sentence_embeddings).squeeze().tolist()
    del sentence_embeddings
    if isinstance(similarity_scores, float):
        similarity_scores = [similarity_scores]
    relevant_indices = [i for i, score in enumerate(similarity_scores) if score >= threshold]
    return relevant_indices, similarity_scores

def clean_and_check_text(webpage_text: str, unwanted_patterns: List[re.Pattern]) -> str:
    if webpage_text == "NA":
        return webpage_text, True

    # Replace two or more newlines with a single newline
    cleaned_text = re.sub(r'\n{2,}', '\n', webpage_text)

    if cleaned_text == "":
        return webpage_text, True

    # Check if any of the unwanted patterns matches the text
    flag = False
    for pattern in unwanted_patterns:
        if pattern.search(cleaned_text):
            flag = True
            break
    return cleaned_text, flag

def process_claims(search_results: Dict[str, Any], database_path: str, prompt_claim_entities_path: str, prompt_entity_relationship_path: str, embed_model="text-embeddings-004"):   
    # Initialize sentence transformer model
    # embeddings_model = SentenceTransformer('all-MiniLM-L12-v2')

    if embed_model == "text-embeddings-004":
        embeddings_model = GeminiAPI(secrets_file="secrets/gemini_keys.json")
        sentence_threshold = 0.55
    else:
        embeddings_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        sentence_threshold = 0.575

    compiled_patterns = load_unwanted_patterns("unwanted_config.json")
    filtered_articles = {}

    for claim_index, claim in enumerate(search_results):
        print(f"\nProcessing Claim ID: {claim_index}\n")
        filtered_articles[claim] = {}
        
        # Get the embedding of the claim
        if embed_model == "text-embeddings-004":
            query_embedding = embeddings_model.get_text_embeddings(batched_text=[claim], task="semantic_similarity")
        else:
            query_embedding = embeddings_model.encode(claim, convert_to_tensor=True)

        for query_index, query in enumerate(search_results[claim].keys()):

            # if embed_model == "text-embeddings-004":
            #     query_embedding = embeddings_model.get_text_embeddings(batched_text=[query], task="semantic_similarity")
            # else:
            #     query_embedding = embeddings_model.encode(query, convert_to_tensor=True)

            for page_index in search_results[claim][query].keys():
                for article_index, article_metadata in enumerate(search_results[claim][query][page_index]):
                    # Skip if the article is already processed
                    if article_metadata["link"] in filtered_articles[claim]:
                        print("Article already processed")
                        continue

                    # Fetch the article save_path from the database
                    search_key = f"{claim_index}_{query_index}_{page_index}_{article_index}"
                    row = retrieve_row_by_key(search_key, database_path)
                    
                    if row:
                        article_path = row['path']
                        print(f"Loading Article: {article_path}")
                        with open(article_path, "r", encoding='utf-8') as f:
                            data = json.load(f)
                            title = data['title'] if 'title' in data else ""
                            article_text, flag = clean_and_check_text(data['text'], compiled_patterns)
                            article_text = f"{title}\n{article_text}"

                            # This article is not useful
                            if flag:
                                print(f"Skipping article: {article_path}")
                                continue
                    else:
                        print("Row not found for key:", search_key)
                        continue

                    if article_text == "NA":
                        print(f"Skipping article: {article_path}, no text")
                        continue
                    
                    # Tokenize sentences
                    sentences = nltk.sent_tokenize(article_text)

                    # Remove large sentences (context length limit is 8192 tokens)
                    # A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).
                    # sentences = [sentence for sentence in sentences if len(sentence.split()) <= 15000]

                    # Remove sentences that are less than 10 words and longer than 2000 characters
                    print(f"Number of Sentences (before filtering): {len(sentences)}")
                    sentences = [sentence for sentence in sentences if len(nltk.tokenize.word_tokenize(sentence)) >= 10 and len(sentence) <= 2000]
                    print(f"Number of Sentences (after filtering): {len(sentences)}")

                    # Skip large or empty articles
                    if len(sentences) > 1000 or len(sentences) == 0:
                        print(f"Skipping article: {article_path}, because #sentences is {len(sentences)}")
                        continue

                    # Step 1: Filter sentences based on similarity
                    if embed_model == "text-embeddings-004":                        
                        gemini_embeddings = embeddings_model.get_text_embeddings(batched_text=sentences, task="semantic_similarity")
                        similarity_scores = util.cos_sim(query_embedding, gemini_embeddings).squeeze().tolist()
                        if isinstance(similarity_scores, float):
                            similarity_scores = [similarity_scores]
                        relevant_sentence_indices = [index for index, score in enumerate(similarity_scores) if score >= sentence_threshold]
                    else:
                        relevant_sentence_indices, similarity_scores = extract_relevant_sentences(
                            sentences,
                            embeddings_model,
                            query_embedding,
                            threshold=sentence_threshold
                        )
                    
                    print(f"Number of Relevant Sentences (threshold={sentence_threshold}): {len(relevant_sentence_indices)}\n")
                    if len(relevant_sentence_indices) == 0:
                        print(f"Skipping article: {article_path}, because 0 similar sentences found")
                        continue

                    # TODO: Use other strategies for segment extraction
                    # article_summary = '\n'.join(sentences[i] for i in relevant_sentence_indices)
                    filtered_articles[claim][article_metadata["link"]] = {
                        "title": article_metadata["title"],
                        "query": query,
                        "scores": [similarity_scores[i] for i in relevant_sentence_indices],
                        "filtered_segments": [sentences[i] for i in relevant_sentence_indices]                        
                    }

        with open("outputs/filtered_results.json", "w") as f:
            json.dump(filtered_articles, f, indent=2)
    
    # Save the knowledge relationships to a JSON file
def main():
    search_results_path = "outputs/search_results.json"
    database_path = "outputs/index.db"
    prompt_claim_entities_path = "prompts/entity_extraction.txt"
    prompt_entity_relationship_path = "prompts/entity_relationship_extraction.txt"
    
    # Load search results
    with open(search_results_path, "r", encoding='utf-8') as fp:
        search_results = json.load(fp)
    
    process_claims(
        search_results=search_results,
        database_path=database_path,
        prompt_claim_entities_path=prompt_claim_entities_path,
        prompt_entity_relationship_path=prompt_entity_relationship_path,
        embed_model="text-embeddings-004"
    )

if __name__ == "__main__":
    main()

# TODO: For each extracted high quality sentence, include its surrounding sentences to provide more context. 
# TODO: CHECK DIFFERENT EMBEDDING COMPARISION - Jaccard Similarity, Cosine Similarity, Euclidean Distance