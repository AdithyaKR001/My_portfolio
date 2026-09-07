import boto3
import time
import os
import json
import concurrent.futures

# ---------- CONFIG ----------
aws_access_key = '<some_access_key>'
aws_secret_key = '<some_secret_key>'
region_name = '<some_region>'  # Change if needed
bucket_name = '<some_bucket_name>'

max_workers = 4  # Number of parallel workers
return_job_id_only = False  # Set to True to only start Textract jobs
# ----------------------------
s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key,
                             aws_secret_access_key=aws_secret_key,
                             region_name=region_name)

def upload_to_s3(local_pdf_path, s3_object_name):
    s3_client.upload_file(local_pdf_path, bucket_name, s3_object_name)
    print(f"✅ Uploaded to s3://{bucket_name}/{s3_object_name}", flush=True)

def start_textract_job(s3_object_name):
    textract_client = boto3.client('textract',
                                   aws_access_key_id=aws_access_key,
                                   aws_secret_access_key=aws_secret_key,
                                   region_name=region_name)

    response = textract_client.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': s3_object_name
            }
        }
    )
    job_id = response['JobId']
    print(f"🚀 Started Textract Job: {job_id}", flush=True)
    return job_id

def wait_for_job_completion(job_id):
    textract_client = boto3.client('textract',
                                   aws_access_key_id=aws_access_key,
                                   aws_secret_access_key=aws_secret_key,
                                   region_name=region_name)

    while True:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            print(f"📌 Job Status: {status}", flush=True)
            return response if status == 'SUCCEEDED' else None
        time.sleep(5)

def extract_text_and_pagecount(initial_response, job_id):
    textract_client = boto3.client('textract',
                                   aws_access_key_id=aws_access_key,
                                   aws_secret_access_key=aws_secret_key,
                                   region_name=region_name)

    text = ''
    page_count = 0

    def process_blocks(blocks):
        nonlocal text, page_count
        for block in blocks:
            if block['BlockType'] == 'LINE':
                text += block['Text'] + '\n'
            elif block['BlockType'] == 'PAGE':
                page_count += 1

    process_blocks(initial_response['Blocks'])

    next_token = initial_response.get('NextToken')
    while next_token:
        response = textract_client.get_document_text_detection(JobId=job_id, NextToken=next_token)
        process_blocks(response['Blocks'])
        next_token = response.get('NextToken')

    return text, page_count

def delete_from_s3(s3_object_name):
    s3_client.delete_object(Bucket=bucket_name, Key=s3_object_name)
    print(f"Deleted {s3_object_name} from S3 bucket {bucket_name}")

def process_pdf_file(local_pdf_path, output_dir):

    s3_object_name = os.path.basename(local_pdf_path)
    base_name = os.path.splitext(s3_object_name)[0]
    jsonl_name = base_name.replace(' ', '_')    
    output_jsonl_path = os.path.join(output_dir, f"{jsonl_name}.jsonl")

    upload_to_s3(local_pdf_path, s3_object_name)
    job_id = start_textract_job(s3_object_name)

    if return_job_id_only:
        return { "file": s3_object_name, "job_id": job_id }

    response = wait_for_job_completion(job_id)

    if response:
        extracted_text, total_pages = extract_text_and_pagecount(response, job_id)
        result = {
            "text": extracted_text,
            "pdf-total-pages": total_pages
        }

        with open(output_jsonl_path, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        delete_from_s3(s3_object_name)
        return { "file": s3_object_name, "status": "success", "pages": total_pages, "output_path": output_jsonl_path }
    else:        
        return { "file": s3_object_name, "status": "failed" }

 

def is_pdf_file(filename):
    return filename.lower().endswith('.pdf')

def ocr_pdf_files(local_pdf_folder, output_dir):
    start_time = time.time()

    pdf_files = [os.path.join(local_pdf_folder, f) for f in os.listdir(local_pdf_folder) if is_pdf_file(f)]
    print(f"📁 Found {len(pdf_files)} PDF(s) in '{local_pdf_folder}'", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdf_file, pdf, output_dir) for pdf in pdf_files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"📝 {result}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total batch OCR time: {elapsed_time:.2f} seconds", flush=True)
