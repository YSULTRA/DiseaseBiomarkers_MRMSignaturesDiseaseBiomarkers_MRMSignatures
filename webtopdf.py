import csv
import os
import requests
import logging
import time
import threading
import pdfkit
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

# ---- CONFIGURATION ----
CSV_FILE = "filtered_file.csv"
OUTPUT_DIR = "3000PDFs"
LOG_DIR = "3000LOGSs"
WEBPAGE_DIR = "3000sWEBPAGE_DOWNLOAD"
MAX_DOWNLOADS = 2000  # Limit the number of DOIs to process
NUM_THREADS = 5  # Number of concurrent threads

# ---- LOG FILES ----
download_log_file = os.path.join(LOG_DIR, "downloaded_log.csv")
not_download_log_file = os.path.join(LOG_DIR, "not_downloaded_log.csv")
last_processed_file = os.path.join(LOG_DIR, "last_processed.txt")

# ---- CREATE DIRECTORIES ----
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WEBPAGE_DIR, exist_ok=True)

# ---- SETUP SELENIUM ----
chrome_options = Options()
chrome_options.binary_location = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service("chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)
driver_lock = threading.Lock()  # Ensure Selenium is accessed safely across threads

# ---- FUNCTION TO WRITE LOGS ----
def write_log(file, data, header):
    file_exists = os.path.isfile(file)
    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # Write header if file does not exist
        writer.writerow(data)

# ---- FUNCTION TO DOWNLOAD PDF ----
def download_pdf(pmid, doi):
    sci_hub_url = f"https://sci-hub.se/{doi}"
    print(f"📄 Processing DOI: {doi} -> {sci_hub_url}")

    try:
        with driver_lock:
            driver.get(sci_hub_url)
            pdf_embed = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#article embed#pdf"))
            )
            pdf_url = pdf_embed.get_attribute("src")

        if pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url

        print(f"✅ PDF found: {pdf_url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        pdf_response = requests.get(pdf_url, headers=headers, stream=True)

        if pdf_response.status_code == 200:
            pdf_path = os.path.join(OUTPUT_DIR, f"{pmid}.pdf")
            with open(pdf_path, "wb") as pdf_file:
                for chunk in pdf_response.iter_content(chunk_size=1024):
                    pdf_file.write(chunk)
            print(f"📥 Downloaded: {pdf_path}")
            write_log(download_log_file, [pmid, doi, "Sci-Hub", pdf_url], ["PMID", "DOI", "Method", "Source"])
            return True
        else:
            raise Exception(f"HTTP {pdf_response.status_code}")
    except Exception as e:
        print(f"❌ Failed to download from Sci-Hub: {e}")
        return False

# ---- FUNCTION TO TRY DIRECT DOI LINK ----
def try_direct_link(pmid, doi):
    direct_url = f"https://doi.org/{doi}"
    print(f"🌐 Trying direct DOI link: {direct_url}")

    try:
        with driver_lock:
            driver.get(direct_url)
            time.sleep(7)  # Allow redirection to complete

        pdf_links = [link.get_attribute("href") for link in driver.find_elements(By.TAG_NAME, "a") if "pdf" in link.get_attribute("href")]

        if pdf_links:
            pdf_url = pdf_links[0]
            headers = {"User-Agent": "Mozilla/5.0"}
            pdf_response = requests.get(pdf_url, headers=headers, stream=True)
            if pdf_response.status_code == 200:
                pdf_path = os.path.join(OUTPUT_DIR, f"{pmid}.pdf")
                with open(pdf_path, "wb") as pdf_file:
                    for chunk in pdf_response.iter_content(chunk_size=1024):
                        pdf_file.write(chunk)
                print(f"📥 Downloaded via DOI: {pdf_path}")
                write_log(download_log_file, [pmid, doi, "Direct DOI", pdf_url], ["PMID", "DOI", "Method", "Source"])
                return True

        print("📄 No direct PDF found, saving webpage as PDF...")

        pdf_save_path = os.path.join(WEBPAGE_DIR, f"{pmid}.pdf")
        pdfkit.from_url(direct_url, pdf_save_path)
        print(f"📝 Webpage saved as PDF: {pdf_save_path}")
        write_log(not_download_log_file, [pmid, doi, "Saved as Webpage PDF", pdf_save_path], ["PMID", "DOI", "Status", "Reason"])
        return False
    except Exception as e:
        print(f"❌ Error accessing DOI: {e}")
        return False

# ---- FUNCTION TO PROCESS DOI ----
def process_doi(row):
    pmid, doi = row.get("PMID", "").strip(), row.get("DOI", "").strip()
    if not pmid or not doi:
        print(f"⚠️ Skipping invalid row: {row}")
        return False

    success = download_pdf(pmid, doi)
    # if not success:
    #     success = try_direct_link(pmid, doi)

    if not success:
        write_log(not_download_log_file, [pmid, doi, "Failed", "No PDF Found"], ["PMID", "DOI", "Status", "Reason"])
    return success

# ---- GET LAST PROCESSED INDEX ----
last_processed_index = 0
if os.path.exists(last_processed_file):
    with open(last_processed_file, "r") as f:
        try:
            last_processed_index = int(f.read().strip())
        except ValueError:
            last_processed_index = 0

download_count = 0

# ---- PROCESS CSV FILE MULTI-THREADED ----
with open(CSV_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    total = min(MAX_DOWNLOADS, len(rows))

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_doi, row): idx for idx, row in enumerate(rows[last_processed_index:total])}

        for future in tqdm(as_completed(futures), total=total, desc="Processing DOIs", unit="DOI"):
            if future.result():
                download_count += 1

            # Update last processed index
            with open(last_processed_file, "w") as f:
                f.write(str(last_processed_index + futures[future]))

print(f"🏁 Total PDFs downloaded: {download_count}")
driver.quit()
