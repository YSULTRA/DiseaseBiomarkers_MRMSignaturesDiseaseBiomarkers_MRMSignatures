# DiseaseBiomarkers_MRMSignatures

This repository contains Python scripts for developing a database of disease biomarkers and their Multiple Reaction Monitoring (MRM) signatures. The project focuses on extracting, processing, and analyzing biomarker data from scientific literature, storing it in MongoDB, and generating insights through exploratory data analysis (EDA).

<img width="1432" height="1055" alt="image" src="https://github.com/user-attachments/assets/a30f2f9d-cad6-447e-b595-28e2a3271fb2" />


## Repository Overview

The repository includes scripts to:
- Download scientific articles as PDFs based on DOIs.
- Extract biomarker data from PDFs and store it in MongoDB.
- Perform EDA on the biomarker dataset to generate visualizations and insights.
- Map diseases to their associated biomarkers in a CSV file.

### Files
- **`eda_code.py`**: Performs exploratory data analysis on biomarker data stored in MongoDB or a JSON file. Generates visualizations (e.g., bar plots, heatmaps, word clouds) and creates a CSV mapping diseases to biomarkers.
- **`LAUNCHER.py`**: Distributes PDF files across subfolders and launches multiple instances of `PROCESS.py` in separate Windows Terminal tabs, each using a different Google API key.
- **`PROCESS.py`**: Processes PDF files to extract biomarker data using Google Gemini API, fetches UniProt details, and stores the results in MongoDB.
- **`webtopdf.py`**: Downloads PDFs from Sci-Hub or direct DOI links based on a CSV file containing PMIDs and DOIs, saving them for processing.
- **`BTP_Report_Biomarker_MRM.pdf`**: Project report documenting the methodology and findings (ensure this file is under 100 MB for GitHub).
- **`.gitignore`**: Excludes large files and sensitive data from version control.

### Large Files (Stored Externally)
The following files are not included in the repository due to GitHub's 100 MB file size limit but are essential for running the scripts:
- **`MasterDB.MasterCollection_2May_2025.json`**: Contains the biomarker dataset for EDA in `eda_code.py`. Download from [insert external storage link, e.g., Google Drive/Dropbox].
- **`BTP_Final_Sheet.xlsx`**: Additional project data (optional). Download from [insert external storage link].

## Prerequisites

### Software
- Python 3.8+
- MongoDB (local or cloud instance)
- Windows Terminal (for `LAUNCHER.py`)
- ChromeDriver and Brave Browser (for `webtopdf.py`)
- wkhtmltopdf (for `webtopdf.py` webpage-to-PDF conversion)

### Python Dependencies
Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn networkx scipy pdfplumber google-generativeai pymongo requests tiktoken tabulate tenacity pdfkit selenium tqdm
