import subprocess
import os
import glob
import shutil
import time
import requests
import traceback
from timeit import default_timer as timer
from .firebase_utils import upload_file_to_gcs, create_folder_in_gcs, upload_to_gcs_folder, download_file_from_firebase, log_ai_service_error
from .textract_utils import ocr_pdf_files
class OCRPipelineError(Exception):
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

def async_ocr_pipeline(policy_key, file_access_key, case_file_id, job_type, input_files, jsonl_map, data):
    try:
        # Step 3a - Create folder in Firebase GCS
        create_folder_in_gcs(f"policies/{policy_key}/{file_access_key}/ocr/")

        # Step 3b - Create folders
        base_dir = f"/app/{file_access_key}"
        input_dir = os.path.join(base_dir, "input")
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Step 3c - Download PDFs
        for file in input_files:
            pdf_url = file['downloadURL']
            pdf_name = file['name']
            print(f"fetch pdf: {pdf_url}")
            download_file_from_firebase(pdf_url,input_dir,pdf_name)

        # Step 3d - OCR PDFs
        ocr_pdf_files(input_dir, output_dir)

        # Step 3e - Upload OCRed .jsonl files to Firebase storage
        upload_to_gcs_folder(f"policies/{policy_key}/{file_access_key}/ocr/", output_dir) #result_dir)

        # Step 3f - Cleanup
        shutil.rmtree(base_dir)

    except Exception as e:
        # Wrap if not already an OCRPipelineError
        print("inside async exception handling", flush=True)
        if not isinstance(e, OCRPipelineError):
            e = OCRPipelineError("Unhandled error in async_ocr_pipeline", context={
                "stage": "async_ocr_pipeline",
                "message": str(e)
            })

        enriched_payload = {
            "message": str(e),
            "jobData": data,
            "stack": traceback.format_exc(),
            "context": getattr(e, "context", {})
        }

        # Log the error directly inside the async pipeline
        log_ai_service_error(
            job_type=job_type,  # or pass as a param if dynamic
            case_file_id=case_file_id,            
            request_data=enriched_payload,
            error=e
        )

def run_olmocr_in_container(encoded_key, policy_key):
    container_dir = f"/app/{encoded_key}"
    input_dir = f"{container_dir}/input"
    output_dir = f"{container_dir}/output"
    pdf_list_path = f"{input_dir}/pdf_list.txt"

    command_inside_container = [
        "python", "-m", "olmocr.pipeline",
        output_dir,
        "--markdown",
        "--pdfs", pdf_list_path,
        "--pages_per_group", "1",
        "--workers", "20",
        "--max_page_error_rate", "2000",
        "--model", "allenai/olmOCR-7B-0225-preview"
    ]

    docker_exec_cmd = ["docker", "exec", "olmocr_container"] + command_inside_container
    start = timer()
    completed = subprocess.run(docker_exec_cmd, capture_output=True, text=True)
    end = timer()

    if completed.returncode != 0:
        raise OCRPipelineError(
        f"Subprocess failed with return code {completed.returncode}",
        context={
            "stage": "subprocess_execution",
            "command": completed.args if hasattr(completed, 'args') else "unknown",
            "returnCode": completed.returncode,
            "stderr": completed.stderr.decode() if completed.stderr else "No stderr",
            "stdout": completed.stdout.decode() if completed.stdout else "No stdout"
        }
    )

    print(f"✅ OCR completed in {end - start:.2f}s. Uploading results...")

    # === Upload JSON results to GCS ===
    results_path = os.path.join(output_dir, "results")
    uploaded_all = True

    for fpath in glob.glob(os.path.join(results_path, "*.jsonl")):
        try:
            gcs_path = f"policies/{policy_key}/OCR/{os.path.basename(fpath)}"
            upload_file_to_gcs(fpath, gcs_path)
        except Exception as e:
            print(f"❌ Failed to upload {fpath}: {e}")
            uploaded_all = False

    # === Cleanup: Remove /app/{encodedKey} directory if upload succeeded ===
    if uploaded_all and os.path.isdir(container_dir):
        try:
            safe_delete_folder(container_dir)
            print(f"🧹 Cleaned up temporary folder: {container_dir}")
        except Exception as e:
            print(f"⚠️ Failed to delete folder {container_dir}: {e}")
    else:
        print(f"⚠️ Skipped cleanup due to failed upload or missing folder: {container_dir}")

def safe_delete_folder(folder_path, retries=3, delay=2):
    for attempt in range(retries):
        try:
            shutil.rmtree(folder_path)
            print(f"🧹 Cleaned up: {folder_path}")
            return True
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} to delete {folder_path} failed: {e}")
            time.sleep(delay)
    print(f"❌ Failed to delete {folder_path} after {retries} attempts.")
    return False

def map_jsonl_files_by_content(jsonl_map, result_dir):
    """
    Rename .jsonl files based on matching PDF filename content.

    Args:
        jsonl_map (dict): Dictionary mapping input PDF names (e.g. 'PDF1.pdf') to desired output JSONL names (e.g. 'PDF1.jsonl')
        result_dir (str): Directory where .jsonl files are generated by olmocr
    """
    for input_pdf, target_jsonl_name in jsonl_map.items():
        matched = False
        for jsonl_file in os.listdir(result_dir):
            if not jsonl_file.endswith(".jsonl"):
                continue

            jsonl_path = os.path.join(result_dir, jsonl_file)
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if input_pdf in line:
                            new_path = os.path.join(result_dir, target_jsonl_name)
                            if jsonl_path != new_path:
                                os.rename(jsonl_path, new_path)
                                print(f"✅ Renamed {jsonl_file} ➜ {target_jsonl_name}")
                            matched = True
                            break  # No need to continue reading this file

                if matched:
                    break  # Stop scanning other .jsonl files for this input_pdf

            except Exception as e:
                raise OCRPipelineError(
                    "Failed in map_jsonl_files_by_content",
                    context={
                        "stage": "map_jsonl_files_by_content",
                        "jsonl_path": jsonl_path
                    }
                ) from e
