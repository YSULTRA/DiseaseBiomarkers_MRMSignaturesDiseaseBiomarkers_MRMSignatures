#!/usr/bin/env python3
import subprocess
import shutil
import time
import logging
import argparse
from pathlib import Path
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------------------------------------------------------------------
# Configuration and Logging Setup
# -------------------------------------------------------------------
def setup_logging() -> None:
    """
    Configure structured logging with a rotating file handler.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler("launcher.log", maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

# -------------------------------------------------------------------
# PDF Distribution Functionality
# -------------------------------------------------------------------
def distribute_pdfs(source_folder: Path, subfolder_paths: List[Path]) -> None:
    """
    Distribute PDF files from the source folder round-robin into target subfolders.
    """
    pdf_files = sorted(source_folder.glob("*.pdf"))
    total_instances = len(subfolder_paths)
    logging.info(f"Found {len(pdf_files)} PDF files in '{source_folder}'.")
    
    for idx, pdf_file in enumerate(pdf_files):
        target_folder = subfolder_paths[idx % total_instances]
        target_file = target_folder / pdf_file.name
        try:
            logging.info(f"Moving '{pdf_file.name}' to '{target_folder}'")
            shutil.move(str(pdf_file), str(target_file))
        except Exception as e:
            logging.error(f"Error moving file {pdf_file.name} to {target_folder}: {e}")

# -------------------------------------------------------------------
# Advanced Subprocess Launcher with Retry
# -------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=15),
       retry=retry_if_exception_type(Exception))
def launch_tab(cmd: str) -> None:
    """
    Launch a Windows Terminal tab using the given command.
    Retries on failure.
    """
    subprocess.Popen(cmd, shell=True)
    logging.info(f"Launched tab with command: {cmd}")

def launch_processes(process_py_path: Path, base_folder: str,
                     api_keys: List[str], api_names: List[str],
                     subfolder_paths: List[Path]) -> None:
    """
    Launch the PROCESS.py script in separate Windows Terminal tabs using provided API keys/names.
    """
    total_instances = len(api_names)
    # Use the last N keys corresponding to the number of API names.
    for i, (api_key, api_name) in enumerate(zip(api_keys[-total_instances:], api_names)):
        pdf_folder = subfolder_paths[i]
        # Construct the Windows Terminal command.
        if i == 0:
            cmd = (
                f'wt cmd /k "python \\"{process_py_path}\\" \\"{api_key}\\" '
                f'\\"{api_name}\\" \\"{i}\\" \\"{total_instances}\\" \\"{pdf_folder}\\""'
            )
        else:
            cmd = (
                f'wt -w 0 new-tab cmd /k "python \\"{process_py_path}\\" \\"{api_key}\\" '
                f'\\"{api_name}\\" \\"{i}\\" \\"{total_instances}\\" \\"{pdf_folder}\\""'
            )
        try:
            launch_tab(cmd)
        except Exception as e:
            logging.error(f"Error launching tab {i}: {e}")
        if i < total_instances - 1:
            logging.info("Sleeping for 10 seconds before launching next tab...")
            time.sleep(10)

# -------------------------------------------------------------------
# Main Functionality
# -------------------------------------------------------------------
def main() -> None:
    """
    Main launcher logic: parse arguments, distribute PDFs, and launch PROCESS.py instances.
    """
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Launcher for PROCESS.py to distribute and process PDFs using multiple API keys."
    )
    parser.add_argument("process_py_path", type=str,
                        help="Full path to PROCESS.py (required)")
    parser.add_argument("base_folder", type=str, nargs="?", default="LAKSHAY_23",
                        help="Name of the folder containing PDF files (optional; default: LAKSHAY_23)")
    args = parser.parse_args()
    
    # Define API keys and names (adjust as needed or read from an external file)
    api_keys = [
        "AIzaSyAs0LwCSfSlECkmcYhVJ8r5sHYKcKNfX6E",
        "AIzaSyBy2kU-4Pg0c13LyW28GObHQi76uElzSFU",
        "AIzaSyBDLDLyYD-d0CtgdXe_S0JPvUWgv9i5LNM",
        "AIzaSyB6xDL9xASwKURMXK3j8jgY_T2SZqqgyl4",
        "AIzaSyAHdzHS9p8foH9YDhSxGfqvczjSwhY1X1M",
        "AIzaSyDy3uI0j1CojY-PpPcExYzbxXxbxcerNWc",
        "AIzaSyDWtFEEOXuS2D1SZNow_wpUgIa_NL0ANbI"
    ]
    api_names = [
        "LAKSHAY_API_1",
        "LAKSHAY_API_2",
        "LAKSHAY_API_3",
        "LAKSHAY_API_4",
        "LAKSHAY_API_5",
        "LAKSHAY_API_6",
        "LAKSHAY_API_7"
    ]
    total_instances = len(api_names)
    
    process_py_path = Path(args.process_py_path).resolve()
    base_dir = process_py_path.parent
    
    # Build the source folder path (expected to be in the same directory as PROCESS.py)
    source_folder = base_dir / args.base_folder
    if not source_folder.exists():
        logging.error(f"Source folder '{source_folder}' does not exist.")
        return
    
    # Create subfolders for each API instance.
    subfolder_paths: List[Path] = []
    for api_name in api_names:
        subfolder = base_dir / f"{args.base_folder}.{api_name.replace(' ', '_')}"
        subfolder.mkdir(parents=True, exist_ok=True)
        subfolder_paths.append(subfolder)
    
    distribute_pdfs(source_folder, subfolder_paths)
    
    logging.info("Mapping of API Keys with API Names:")
    for api_key, api_name in zip(api_keys[-total_instances:], api_names):
        logging.info(f"{api_name}: {api_key}")
    
    launch_processes(process_py_path, args.base_folder, api_keys, api_names, subfolder_paths)
    logging.info("Launched all tabs.")

if __name__ == "__main__":
    main()
