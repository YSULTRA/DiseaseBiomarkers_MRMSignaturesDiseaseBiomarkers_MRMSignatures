#!/usr/bin/env python3
import os
import glob
import time
import re
import sys
import json
import pdfplumber
import google.generativeai as genai
from pymongo import MongoClient
import urllib.parse
import requests
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------
# Configuration Map
# -------------------------------------------------------------------
CONFIG = {
    "MONGODB_URI": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
    "MONGODB_DB": os.getenv("MONGODB_DB", "BiomarkerDB"),
    "MONGODB_COLLECTION_1": os.getenv("MONGODB_COLLECTION_1", "LAKSHAY_25"),
    "MONGODB_COLLECTION_2": os.getenv("MONGODB_COLLECTION_2", "LAKSHAY_25.1"),
    "SERVER_SELECTION_TIMEOUT_MS": 5000,
    "SOCKET_TIMEOUT_MS": 600000,
    "CONNECT_TIMEOUT_MS": 30000,
    "MAX_POOL_SIZE": 50,
    "WAIT_QUEUE_TIMEOUT_MS": 10000,
    "TOKEN_LIMIT": 999999,  # Maximum allowed tokens
    "MODEL_NAME": "gemini-2.0-flash-exp",
    "MAX_OUTPUT_TOKENS": 8192,
    "API_TEMPERATURE": 1,
    "API_TOP_P": 0.95,
    "API_TOP_K": 50,
    "CONCURRENCY": 4,
    "GEN_API_MAX_CALLS_PER_MINUTE": 20,
}

# -------------------------------------------------------------------
# Global Variables for MongoDB connection
# -------------------------------------------------------------------
client = None
db = None
collection = None

# -------------------------------------------------------------------
# Initialize DB and Gemini API
# -------------------------------------------------------------------
def init_db_and_api(api_key):
    global client, db, collection
    client = MongoClient(CONFIG["MONGODB_URI"],
                         serverSelectionTimeoutMS=CONFIG["SERVER_SELECTION_TIMEOUT_MS"],
                         socketTimeoutMS=CONFIG["SOCKET_TIMEOUT_MS"],
                         connectTimeoutMS=CONFIG["CONNECT_TIMEOUT_MS"],
                         maxPoolSize=CONFIG["MAX_POOL_SIZE"],
                         waitQueueTimeoutMS=CONFIG["WAIT_QUEUE_TIMEOUT_MS"],
                         retryWrites=True)
    db = client[CONFIG["MONGODB_DB"]]
    collection = db[CONFIG["MONGODB_COLLECTION_1"]]
    genai.configure(api_key=api_key)
    print("Initialized MongoDB and Gemini API.")
    return client, db, collection

def ensure_mongo_connection():
    global client, db, collection
    try:
        if client is None or client.admin.command('ping') is None:
            client = MongoClient(CONFIG["MONGODB_URI"],
                                 serverSelectionTimeoutMS=CONFIG["SERVER_SELECTION_TIMEOUT_MS"],
                                 socketTimeoutMS=CONFIG["SOCKET_TIMEOUT_MS"],
                                 connectTimeoutMS=CONFIG["CONNECT_TIMEOUT_MS"],
                                 maxPoolSize=CONFIG["MAX_POOL_SIZE"],
                                 waitQueueTimeoutMS=CONFIG["WAIT_QUEUE_TIMEOUT_MS"],
                                 retryWrites=True)
            db = client[CONFIG["MONGODB_DB"]]
            collection = db[CONFIG["MONGODB_COLLECTION_2"]]
            print("MongoDB connection re-established successfully.")
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        time.sleep(5)
        ensure_mongo_connection()

# -------------------------------------------------------------------
# PDF Text Extraction and Advanced Processing
# -------------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Convert any non-string objects to string
                text += str(page_text) + "\n"
    text = re.sub(r'(?i)(References|Bibliography)(.*)', '', text, flags=re.DOTALL)
    return text


def advanced_text_processing(text):
    """Split text into paragraphs and remove excessive whitespace."""
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    return "\n\n".join(paragraphs)

# -------------------------------------------------------------------
# Helper: Normalize Greek Characters
# -------------------------------------------------------------------
def normalize_protein_name(name):
    greek_to_english = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
        'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
        'Ε': 'Epsilon', 'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta',
        'Ι': 'Iota', 'Κ': 'Kappa', 'Λ': 'Lambda', 'Μ': 'Mu',
        'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron', 'Π': 'Pi',
        'Ρ': 'rho', 'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon',
        'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega'
    }
    return "".join(greek_to_english.get(char, char) for char in name)

# -------------------------------------------------------------------
# UniProt Fetching Function (Cached & Retried)
# -------------------------------------------------------------------
@lru_cache(maxsize=256)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(Exception))
def fetch_top_proteins(protein_name):
    query = f'"{protein_name}" AND (organism_id:9606)'
    encoded_query = urllib.parse.quote(query)
    url = f'https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&format=tsv&columns=entry_name,organism,protein_name'
    print("Querying UniProt with URL:", url)
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            headers = lines[0].split('\t')
            first_result = lines[1].split('\t')
            result = dict(zip(headers, first_result))
            print("Fetched UniProt details for protein:", protein_name)
            print("UniProt result:")
            print(tabulate(result.items(), headers=["Field", "Value"], tablefmt="grid"))
            return result
        else:
            print(f"No response found for the {protein_name}")
            return None
    else:
        print(f"Failed to fetch data for {protein_name}. Status code: {response.status_code}")
        return {"error": f"Failed to fetch data. Status code: {response.status_code}"}

# -------------------------------------------------------------------
# Gemini Prompt to Fix Protein Name
# -------------------------------------------------------------------
def fix_protein_name_via_gemini(bad_name):
    prompt = f"Given the protein name '{bad_name}', suggest a corrected protein name that is more likely to match UniProt entries."
    print("Calling Gemini to fix protein name for:", bad_name)
    model = genai.GenerativeModel(
        model_name=CONFIG["MODEL_NAME"],
        generation_config={
            "temperature": CONFIG["API_TEMPERATURE"],
            "top_p": CONFIG["API_TOP_P"],
            "top_k": CONFIG["API_TOP_K"],
            "max_output_tokens": CONFIG["MAX_OUTPUT_TOKENS"],
            "response_mime_type": "application/json",
        }
    )
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    try:
        data = json.loads(response.text)
        if isinstance(data, list):
            data = data[0]  # Take the first element if it's a list
        corrected_name = data.get("output", "").strip()
        print("Gemini corrected protein name:", corrected_name)
        if corrected_name:
            return corrected_name
    except Exception as e:
        print("Error parsing Gemini response:", e)
    return bad_name

# -------------------------------------------------------------------
# Token Counting and Prompt Preparation
# -------------------------------------------------------------------
def count_tokens(text, model_name="gemini-2.0-flash-exp"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def truncate_text_to_token_limit(static_prompt, pdf_text,
                                 token_limit=CONFIG["TOKEN_LIMIT"],
                                 model_name=CONFIG["MODEL_NAME"]):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    static_token_count = count_tokens(static_prompt, model_name)
    pdf_tokens = encoding.encode(pdf_text)
    if static_token_count + len(pdf_tokens) > token_limit:
        allowed_pdf_tokens = token_limit - static_token_count
        pdf_tokens = pdf_tokens[:allowed_pdf_tokens]
        pdf_text = encoding.decode(pdf_tokens)
    return pdf_text

def split_text_into_chunks(text, chunk_token_limit, model_name=CONFIG["MODEL_NAME"]):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_token_limit):
        chunk_tokens = tokens[i:i+chunk_token_limit]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def build_prompt(static_prompt, pdf_text, token_limit=CONFIG["TOKEN_LIMIT"]):
    processed_text = advanced_text_processing(pdf_text)
    processed_text = truncate_text_to_token_limit(static_prompt, processed_text, token_limit)
    try:
        encoding = tiktoken.encoding_for_model(CONFIG["MODEL_NAME"])
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    static_token_count = count_tokens(static_prompt, CONFIG["MODEL_NAME"])
    allowed_tokens = token_limit - static_token_count
    pdf_tokens = encoding.encode(processed_text)
    if len(pdf_tokens) > allowed_tokens:
        print(f"PDF text exceeds allowed token count ({len(pdf_tokens)} > {allowed_tokens}). Splitting into chunks.")
        chunks = split_text_into_chunks(processed_text, allowed_tokens, CONFIG["MODEL_NAME"])
        prompts = [static_prompt + "\n\n" + chunk for chunk in chunks]
    else:
        prompts = [static_prompt + "\n\n" + processed_text]
    print(f"Final prompt token count (first chunk): {count_tokens(prompts[0], CONFIG['MODEL_NAME'])}")
    return prompts

def prepare_static_prompt():
    return """
Optimized RAG Prompt for Biomarker Data Extraction
Context:
You are an advanced AI system designed to analyze and extract specific information about protein biomarkers from research articles. Your task is to process scientific content, including abstracts and supporting text, and extract detailed, structured, and accurate information for database storage. Use the provided context to understand the task requirements fully.

Task:
Extract and organize the following information from each research article:

Biomarker Information:
Protein Name(s): Extract all protein biomarkers mentioned in the abstract or text. Ensure the names are accurate and refer to proteins, not genes.
UniProt ID(s): Retrieve the UniProt ID(s) for each identified protein biomarker. If unavailable, leave the field as null.

Study and Source Information:
Disease Name(s): Identify all diseases associated with the protein biomarkers.
Source Material: Extract the biological source of the biomarker (e.g., serum, blood, saliva, urine, amniotic fluid, cerebrospinal fluid, etc.).
Organism: Always verify the organism as Homo sapiens (humans). If the study involves a different organism, explicitly note it.
Technique Used: Extract all techniques used to identify or analyze the biomarkers (e.g., MS/MS, ELISA, Western Blot, etc.).

Output Format:
Return the information in JSON format for easy storage and processing. Ensure all fields are present, even if the value is null.

{
  "Biomarkers": [
    {
      "ProteinName": "Fetch from uniprot_info -> proteinDescription -> recommendedName -> fullName -> value",
      "UniProtID": "Fetch from uniprot_info -> primaryAccession",
      "EntryName": "Fetch from uniprot_info -> uniProtkbId",
      "GeneNames": "Fetch from uniprot_info -> genes[0] -> geneName -> value (join multiple if needed)",
      "Organism": "Fetch from uniprot_info -> organism -> scientificName"
    }
  ],
  "DiseaseName": "List of diseases associated with the biomarker(s)",
  "SourceMaterial": "Biological source of the biomarker",
  "Organism": "mention the organism if specified",
  "TechniqueUsed": "List of techniques used in the study"
}

Key Instructions for Accuracy and Precision:
- Extract all biomarkers; if multiple protein biomarkers are mentioned, list each in the "Biomarkers" array.
- Verify protein names to ensure they refer to proteins, not genes or other molecules.
- Include all techniques if multiple are used.
- Handle missing data: set fields to null if not explicitly mentioned.
- Default organism to Homo sapiens unless otherwise stated.
- Extract all diseases linked to the biomarkers.
- Remove any biomarker with null or empty ProteinName.
    """

def check_missing_or_null_fields(response_data):
    missing_or_null_fields = {}
    if isinstance(response_data, list):
        response_data = response_data[0]
    required_keys = ["Biomarkers", "DiseaseName", "SourceMaterial", "Organism", "TechniqueUsed"]
    for key in required_keys:
        if key not in response_data or response_data[key] in (None, "", []):
            missing_or_null_fields[key] = True
        else:
            missing_or_null_fields[key] = False
    if "Biomarkers" in response_data:
        if not response_data["Biomarkers"]:
            missing_or_null_fields["Biomarkers"] = True
        else:
            for biomarker in response_data["Biomarkers"]:
                if not biomarker.get("ProteinName"):
                    missing_or_null_fields["Biomarkers"] = True
    return missing_or_null_fields

# -------------------------------------------------------------------
# Processing the PDF and Gemini Interaction (with UniProt Retry)
# -------------------------------------------------------------------
def get_uniprot_details_via_gemini(candidate):
    """
    As a final fallback, ask Gemini to provide UniProt details in JSON format for the given candidate protein.
    The expected JSON should include keys: 'protein_name' (the full standardized protein name),
    'entry_name' (the unique identifier), 'Gene Names', and 'organism'.
    Then, use the Gemini-provided protein name to perform a robust UniProt lookup.
    If the UniProt result is valid and the names match (or are similar), return the UniProt details.
    Otherwise, return None.
    """
    prompt = (f"Provide UniProt details in JSON format for the protein '{candidate}' "
              f"for Homo sapiens. The JSON should include at least the following keys: "
              f"'protein_name' (the full standardized protein name), 'entry_name' (the unique identifier), "
              f"'Gene Names', and 'organism'.")
    print(f"Calling Gemini to fetch UniProt details for '{candidate}' directly.")
    model = genai.GenerativeModel(
        model_name=CONFIG["MODEL_NAME"],
        generation_config={
            "temperature": CONFIG["API_TEMPERATURE"],
            "top_p": CONFIG["API_TOP_P"],
            "top_k": CONFIG["API_TOP_K"],
            "max_output_tokens": CONFIG["MAX_OUTPUT_TOKENS"],
            "response_mime_type": "application/json",
        }
    )
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    try:
        details = json.loads(response.text)
        if isinstance(details, list):
            details = details[0]
        # Check if Gemini response provides the required keys
        if details.get("protein_name") and details.get("entry_name"):
            gemini_protein_name = details.get("protein_name").strip().lower()
            print(f"Gemini provided UniProt details for '{candidate}':")
            print(tabulate(details.items(), headers=["Field", "Value"], tablefmt="grid"))
            # Now perform a robust UniProt lookup using the Gemini-provided protein name.
            uni_details = fetch_top_proteins(gemini_protein_name)
            if uni_details:
                uni_protein_name = (uni_details.get("protein_name") or uni_details.get("Protein names") or "").strip().lower()
                # Check if the names match (or at least one is a substring of the other)
                if uni_protein_name and (gemini_protein_name in uni_protein_name or uni_protein_name in gemini_protein_name):
                    print("Robust UniProt details found using the Gemini-provided protein name:")
                    print(tabulate(uni_details.items(), headers=["Field", "Value"], tablefmt="grid"))
                    return uni_details
                else:
                    print("UniProt lookup using Gemini-provided name did not return matching details; prioritizing UniProt details if available.")
                    # If uni_details exists even though the names are not a perfect match, you can decide whether to return them
                    # Here we choose to return them if the uni_protein_name is non-empty.
                    if uni_protein_name:
                        return uni_details
                    else:
                        print("UniProt details are insufficient; returning None.")
                        return None
            else:
                print("UniProt lookup using Gemini-provided name failed; returning None.")
                return None
        else:
            print(f"Gemini fallback did not return sufficient details for '{candidate}'.")
            return None
    except Exception as e:
        print(f"Error parsing Gemini fallback response for '{candidate}': {e}")
        return None



def fix_missing_fields_via_gemini(response_data, pdf_text):
    """
    If any of the extra fields (Biomarkers, DiseaseName, SourceMaterial, Organism, TechniqueUsed)
    are missing or null, call Gemini with a prompt asking to extract those fields solely from the provided PDF text.
    """
    missing_keys = []
    for key in ["Biomarkers", "DiseaseName", "SourceMaterial", "Organism", "TechniqueUsed"]:
        if key not in response_data or not response_data[key]:
            missing_keys.append(key)
    if missing_keys:
        # Create a static prompt portion without the PDF text
        static_prompt = (
            f"The following fields are missing or empty: {', '.join(missing_keys)}. "
            "Please extract and provide complete, accurate details for these fields solely from the text provided."
        )
        # Build the full prompt by appending the PDF text and instructions for JSON formatting.
        full_prompt = (
            f"{static_prompt}\n\n"
            f"{pdf_text}\n\n"
            "Return the result in JSON format."
        )
        print("📣 Calling Gemini to fetch missing extra fields with prompt:")
        print(static_prompt)
        model = genai.GenerativeModel(
            model_name=CONFIG["MODEL_NAME"],
            generation_config={
                "temperature": CONFIG["API_TEMPERATURE"],
                "top_p": CONFIG["API_TOP_P"],
                "top_k": CONFIG["API_TOP_K"],
                "max_output_tokens": CONFIG["MAX_OUTPUT_TOKENS"],
                "response_mime_type": "application/json",
            }
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(full_prompt)
        try:
            gemini_data = json.loads(response.text)
            # If Gemini returns a list, take the first element
            if isinstance(gemini_data, list):
                gemini_data = gemini_data[0]
            # Update missing keys if found in Gemini's response
            for key in missing_keys:
                if gemini_data.get(key):
                    response_data[key] = gemini_data.get(key)
                    print(f"✅ Updated '{key}' via Gemini fallback: {response_data[key]}")
                else:
                    # Optionally, assign a default value if still missing
                    response_data[key] = "Not available"
                    print(f"⚠️ '{key}' still missing. Setting default value: 'Not available'")
        except Exception as e:
            print("❌ Error fetching missing fields via Gemini:", e)
    return response_data

    
def process_pdf(file_path, collection):
    print(f"🔍 Processing PDF: {os.path.basename(file_path)}")
    pdf_text = extract_text_from_pdf(file_path)
    prompts = build_prompt(prepare_static_prompt(), pdf_text)
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 50,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )
    
    combined_biomarkers = []
    responseOld = None
    base_response = {}

    for prompt in prompts:
        chat_session = model.start_chat(history=[])
        max_retries = 2
        retry_count = 0
        all_fields_filled = False
        while retry_count < max_retries and not all_fields_filled:
            time.sleep(5)
            response = chat_session.send_message(prompt)
            if isinstance(response, dict) and "error" in response:
                error_message = response["error"].get("message", "")
                if "Resource has been exhausted" in error_message:
                    print(f"⏳ Attempt {retry_count+1}: Quota exhausted. Retrying after delay... 🔄")
                    time.sleep(60)
                    continue

            try:
                response_data = json.loads(response.text)
                responseOld = response_data
                if isinstance(response_data, list):
                    response_data = response_data[0]

                print(f"📝 Attempt {retry_count+1} Response:")
                print(json.dumps(response_data, indent=4))

                missing_or_null_fields = check_missing_or_null_fields(response_data)
                # Remove the check for UniProtID so that we only focus on required fields like ProteinName
                missing_or_null_fields = {k: v for k, v in missing_or_null_fields.items() if k != "UniProtID"}

                if not any(missing_or_null_fields.values()):
                    candidates_for_chunk = []
                    if "Biomarkers" in response_data:
                        for biomarker in response_data["Biomarkers"]:
                            original_protein = biomarker.get("ProteinName", "")
                            if original_protein:
                                cand_list = re.split(r'[,;/]', original_protein)
                                for candidate in cand_list:
                                    candidate = candidate.strip()
                                    if candidate:
                                        norm_candidate = normalize_protein_name(candidate)
                                        print(f"🔬 Processing candidate protein: {candidate} (normalized: {norm_candidate})")
                                        details = None
                                        for attempt in range(5):
                                            details = fetch_top_proteins(norm_candidate)
                                            if details is not None:
                                                print(f"✅ UniProt details found for '{norm_candidate}' on attempt {attempt+1}")
                                                break
                                            else:
                                                print(f"❌ Attempt {attempt+1} failed for candidate: {norm_candidate}")
                                                time.sleep(2 ** attempt)
                                        if details is None:
                                            fixed_candidate = fix_protein_name_via_gemini(candidate)
                                            print(f"🤖 Using Gemini-corrected protein name: {fixed_candidate}")
                                            norm_fixed = normalize_protein_name(fixed_candidate)
                                            details = fetch_top_proteins(norm_fixed)
                                            if details is None:
                                                details = get_uniprot_details_via_gemini(candidate)
                                        if details is not None:
                                            standard_name = details.get("protein_name") or details.get("Protein names") or candidate
                                            new_entry = biomarker.copy()
                                            new_entry["ProteinName"] = standard_name
                                            new_entry["UniProtID"] = details.get("Entry")
                                            new_entry["EntryName"] = details.get("Entry Name")
                                            new_entry["GeneNames"] = details.get("Gene Names") or details.get("gene_name")
                                            new_entry["Organism"] = details.get("Organism")
                                            print(f"🔄 UniProt details updated for {candidate} (standardized as {standard_name}):")
                                            table_data = list(details.items())
                                            print(tabulate(table_data, headers=["Field", "Value"], tablefmt="grid"))
                                            candidates_for_chunk.append(new_entry)
                                        else:
                                            print(f"🚫 Candidate '{candidate}' failed to fetch UniProt details after 5 attempts and will be removed.")
                    if not candidates_for_chunk:
                        # Additional fallback: try Gemini on all candidates from this chunk
                        print("🔄 No candidate biomarkers with UniProt details found in this chunk; attempting Gemini fallback for all candidates.")
                        if "Biomarkers" in response_data:
                            for biomarker in response_data["Biomarkers"]:
                                original_protein = biomarker.get("ProteinName", "")
                                if original_protein:
                                    for candidate in re.split(r'[,;/]', original_protein):
                                        candidate = candidate.strip()
                                        if candidate:
                                            fixed_candidate = fix_protein_name_via_gemini(candidate)
                                            details = fetch_top_proteins(fixed_candidate)
                                            if details is not None:
                                                new_name = details.get("protein_name") or details.get("Protein names") or fixed_candidate
                                                new_entry = biomarker.copy()
                                                new_entry["ProteinName"] = new_name
                                                new_entry["UniProtID"] = details.get("Entry") or details.get("entry_name")
                                                new_entry["EntryName"] = details.get("Entry Name") or details.get("entry_name")
                                                new_entry["GeneNames"] = details.get("Gene Names") or details.get("protein_name")
                                                new_entry["Organism"] = details.get("Organism") or details.get("organism")
                                                print(f"🤖 Gemini fallback updated details for {candidate} (standardized as {new_name}):")
                                                print(tabulate(list(details.items()), headers=["Field", "Value"], tablefmt="grid"))
                                                candidates_for_chunk.append(new_entry)
                        if candidates_for_chunk:
                            combined_biomarkers.extend(candidates_for_chunk)
                            all_fields_filled = True
                        else:
                            print("🔁 Still no candidate biomarkers with UniProt details found after Gemini fallback; retrying extraction...")
                            retry_count += 1
                            time.sleep(5 * (2 ** retry_count))
                            prompt += "\n\nAdditional Instructions:\nEnsure each protein biomarker includes complete UniProt details."
                    else:
                        combined_biomarkers.extend(candidates_for_chunk)
                        all_fields_filled = True
                else:
                    print(f"⚠️ Missing/Null Fields Detected: {missing_or_null_fields}")
                    missing_fields_message = ""
                    for field, missing in missing_or_null_fields.items():
                        if missing:
                            missing_fields_message += f"- {field}\n"
                    print(missing_fields_message)
                    retry_count += 1
                    time.sleep(5 * (2 ** retry_count))
                    prompt += f"\n\nAdditional Instructions:\nPlease ensure the following fields are filled:\n{missing_fields_message}"
            except json.JSONDecodeError:
                print("❌ Error: Response is not valid JSON.")
                print(response.text)
                retry_count += 1
                time.sleep(5 * (2 ** retry_count))
        
        # After max attempts, if not all fields are filled, process the last JSON response:
        if not all_fields_filled:
            print("⏱ Max attempts reached. Filtering out biomarkers with null ProteinName from the last JSON response.")
            if isinstance(responseOld, dict) and "Biomarkers" in responseOld:
                filtered_biomarkers = [b for b in responseOld["Biomarkers"] if b.get("ProteinName")]
                if filtered_biomarkers:
                    candidates_for_chunk = []
                    for biomarker in filtered_biomarkers:
                        original_protein = biomarker.get("ProteinName", "")
                        cand_list = re.split(r'[,;/]', original_protein)
                        for candidate in cand_list:
                            candidate = candidate.strip()
                            if candidate:
                                norm_candidate = normalize_protein_name(candidate)
                                print(f"🔬 Processing candidate protein (post-max attempts): {candidate} (normalized: {norm_candidate})")
                                details = None
                                for attempt in range(5):
                                    details = fetch_top_proteins(norm_candidate)
                                    if details is not None:
                                        print(f"✅ UniProt details found for '{norm_candidate}' on attempt {attempt+1}")
                                        break
                                    else:
                                        print(f"❌ Attempt {attempt+1} failed for candidate: {norm_candidate}")
                                        time.sleep(2 ** attempt)
                                if details is None:
                                    fixed_candidate = fix_protein_name_via_gemini(candidate)
                                    print(f"🤖 Using Gemini-corrected protein name: {fixed_candidate}")
                                    norm_fixed = normalize_protein_name(fixed_candidate)
                                    details = fetch_top_proteins(norm_fixed)
                                    if details is None:
                                        details = get_uniprot_details_via_gemini(candidate)
                                if details is not None:
                                    standard_name = details.get("protein_name") or details.get("Protein names") or candidate
                                    new_entry = biomarker.copy()
                                    new_entry["ProteinName"] = standard_name
                                    new_entry["UniProtID"] = details.get("Entry")
                                    new_entry["EntryName"] = details.get("Entry Name")
                                    new_entry["GeneNames"] = details.get("Gene Names") or details.get("gene_name")
                                    new_entry["Organism"] = details.get("Organism")
                                    candidates_for_chunk.append(new_entry)
                                else:
                                    print(f"🚫 Candidate '{candidate}' failed to fetch UniProt details even after max attempts filtering.")
                    if candidates_for_chunk:
                        combined_biomarkers.extend(candidates_for_chunk)
                    else:
                        print("❌ No valid biomarkers with non-null ProteinName found after max attempts.")
            # Also, check and fix missing required fields in the base response
            if isinstance(responseOld, dict):
                base_response = responseOld.copy()
            else:
                base_response = {}
            required_fields = ["DiseaseName", "TechniqueUsed", "SourceMaterial", "Organism"]
            missing_fields = [field for field in required_fields if not base_response.get(field)]
            if missing_fields:
                print(f"⚠️ Missing required fields after max attempts: {missing_fields}. Calling fix_missing_fields_via_gemini() to complete data.")
                base_response = fix_missing_fields_via_gemini(base_response, pdf_text)
    
    # Fallback: if no candidates were found from prompts but base_response has Biomarkers,
    # then process those biomarkers through the UniProt verification pipeline.
        # Fallback: if no candidates were found from prompts but base_response has Biomarkers,
    # then process those biomarkers through the UniProt verification pipeline.
        # Fallback: if no candidates were found from prompts but base_response has Biomarkers,
    # then process those biomarkers through the UniProt verification pipeline.
    if not combined_biomarkers and base_response.get("Biomarkers"):
        print("✅ Using fallback Gemini response biomarkers for UniProt verification.")
        fallback_candidates = []
        biomarkers_field = base_response["Biomarkers"]
        # If the Biomarkers field is a string, wrap it in a list.
        if isinstance(biomarkers_field, str):
            biomarkers_field = [biomarkers_field]
        for biomarker in biomarkers_field:
            # If the biomarker is a dict, use its 'ProteinName' field; if it's a string, treat it as the protein name.
            if isinstance(biomarker, dict):
                original_protein = biomarker.get("ProteinName", "")
            elif isinstance(biomarker, str):
                original_protein = biomarker
            else:
                continue

            if original_protein:
                # Check if delimiters exist; if not, treat the whole string as one candidate.
                if any(delim in original_protein for delim in [',', ';', '/']):
                    cand_list = re.split(r'[,;/]', original_protein)
                else:
                    cand_list = [original_protein]
                for candidate in cand_list:
                    candidate = candidate.strip()
                    if candidate:
                        norm_candidate = normalize_protein_name(candidate)
                        print(f"🔬 Processing fallback candidate protein: {candidate} (normalized: {norm_candidate})")
                        details = None
                        for attempt in range(5):
                            details = fetch_top_proteins(norm_candidate)
                            if details is not None:
                                print(f"✅ UniProt details found for fallback '{norm_candidate}' on attempt {attempt+1}")
                                break
                            else:
                                print(f"❌ Fallback attempt {attempt+1} failed for candidate: {norm_candidate}")
                                time.sleep(2 ** attempt)
                        if details is None:
                            fixed_candidate = fix_protein_name_via_gemini(candidate)
                            print(f"🤖 Using Gemini-corrected protein name for fallback: {fixed_candidate}")
                            norm_fixed = normalize_protein_name(fixed_candidate)
                            details = fetch_top_proteins(norm_fixed)
                            if details is None:
                                details = get_uniprot_details_via_gemini(candidate)
                        if details is not None:
                            standard_name = details.get("protein_name") or details.get("Protein names") or candidate
                            new_entry = {"ProteinName": standard_name}
                            new_entry["UniProtID"] = details.get("Entry")
                            new_entry["EntryName"] = details.get("Entry Name")
                            new_entry["GeneNames"] = details.get("Gene Names") or details.get("gene_name")
                            new_entry["Organism"] = details.get("Organism")
                            fallback_candidates.append(new_entry)
                        else:
                            print(f"🚫 Fallback candidate '{candidate}' failed to fetch UniProt details.")
        if fallback_candidates:
            combined_biomarkers = fallback_candidates



    if combined_biomarkers:
        # Remove duplicate candidates based on ProteinName and filter out entries with missing names or IDs
        unique_candidates = {cand["ProteinName"]: cand for cand in combined_biomarkers if cand.get("ProteinName")}
        final_biomarkers = [cand for cand in unique_candidates.values() if cand.get("ProteinName") and cand.get("UniProtID")]

        # Use the original Gemini response (responseOld) as a base if available
        if isinstance(responseOld, list) and responseOld:
            base_response = responseOld[0]
        elif isinstance(responseOld, dict):
            base_response = responseOld.copy()
        else:
            base_response = {}

        # Check if important fields are missing and fix if needed
        required_fields = ["DiseaseName", "TechniqueUsed", "SourceMaterial", "Organism"]
        missing_fields = [field for field in required_fields if not base_response.get(field)]
        if missing_fields:
            print(f"⚠️ Missing fields detected: {missing_fields}. Calling fix_missing_fields_via_gemini() to complete data.")
            base_response = fix_missing_fields_via_gemini(base_response, pdf_text)

        # Overwrite Biomarkers field and set PMID
        base_response["Biomarkers"] = final_biomarkers
        base_response["PMID"] = os.path.basename(file_path).split('.')[0]
        final_response = base_response

        print("\n✅ Combined Response Data with Biomarkers:")
        print(json.dumps(final_response, indent=4))

        if final_response["Biomarkers"]:
            inserted_id = collection.insert_one(final_response).inserted_id
            print(f"👍 Inserted Entry in MongoDB with ID: {inserted_id}")
        else:
            print("❌ No valid biomarkers found. No entry inserted.")
    else:
        print("❌ No candidate biomarkers found across all chunks. No entry inserted.")

# -------------------------------------------------------------------
# Process All PDFs and Cleanup Functions
# -------------------------------------------------------------------
def process_all_pdfs(collection, instance_id, total_instances, api_name, folder_path, api_key):
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True))
    print(f"Instance {instance_id} sees the following files: {[os.path.basename(f) for f in pdf_files]}")
    
    for file_path in pdf_files:
        print(f"Instance {instance_id}: Processing PDF '{os.path.basename(file_path)}' using {api_name}, API key: {api_key} on folder: {folder_path}")
        try:
            process_pdf(file_path, collection)
            log_file_path = os.path.join(folder_path, "pdf_processing_log.txt")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Successfully Processed: {os.path.basename(file_path)}\n")
        except Exception as e:
            print(f"Instance {instance_id}: Error processing {file_path}: {e}")
            print(f"Instance {instance_id}: Sleeping for 60 seconds before next attempt")
            time.sleep(60)

def remove_processed_files(folder_path):
    log_file_path = os.path.join(folder_path, "pdf_processing_log.txt")
    processed_files = set()
    
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                match = re.search(r'Successfully Processed: (\d+\.pdf)', line)
                if match:
                    processed_files.add(match.group(1))
                    
        for pdf_file in processed_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    print(f"Removed processed file: {pdf_file}")
                except Exception as e:
                    print(f"Error removing file {pdf_file}: {e}")
            else:
                print(f"File not found for removal: {pdf_file}")
        try:
            os.remove(log_file_path)
            print("Log file deleted.")
        except Exception as e:
            print(f"Error deleting log file: {e}")

def rerun_until_empty(collection, instance_id, total_instances, api_name, folder_path, api_key):
    while True:
        ensure_mongo_connection()
        process_all_pdfs(collection, instance_id, total_instances, api_name, folder_path, api_key)
        remove_processed_files(folder_path)
        
        remaining_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True))
        print(folder_path)
        if not remaining_files:
            print("All PDF files processed and deleted. Folder is empty.")
            break
        else:
            print(f"Remaining files: {[os.path.basename(f) for f in remaining_files]}")
            print("Re-running the process...")
        time.sleep(60)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python process.py <api_key> <api_name> <instance_id> <total_instances> <folder_path>")
        sys.exit(1)
    
    print("BEFORE:", "(global FOLDER_PATH is not set)")
    folder_path = sys.argv[5]
    print("AFTER:", folder_path)
    
    api_key = sys.argv[1]
    api_name = sys.argv[2]
    
    try:
        instance_id = int(sys.argv[3])
        total_instances = int(sys.argv[4])
    except ValueError:
        print("instance_id and total_instances must be integers.")
        sys.exit(1)
    
    print(f"Started continuous processing using {api_name} (instance {instance_id} of {total_instances}) on folder {folder_path}...")
    init_db_and_api(api_key)
    rerun_until_empty(collection, instance_id, total_instances, api_name, folder_path, api_key)
