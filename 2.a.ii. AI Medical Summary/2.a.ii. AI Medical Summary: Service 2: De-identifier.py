from .firebase_utils import read_file_from_firebase, save_deidentified_file_to_gcs, log_ai_service_error
from google.cloud import dlp_v2
from google.cloud.dlp_v2.types import DeidentifyContentRequest
import json, traceback

class DeIdentifierPipelineError(Exception):
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

EXCLUDED_INFO_TYPES = {
    "DATE", "AGE", "BLOOD_TYPE", "DATE_OF_BIRTH", "DEMOGRAPHIC_DATA",
    "ETHNIC_GROUP", "GENDER", "MARITAL_STATUS", "MEDICAL_DATA",
    "MEDICAL_TERM", "SEXUAL_ORIENTATION"
}

SELECTED_INFO_TYPES = [
    {"name": "FEMALE_NAME"},
    {"name": "MALE_NAME"},
    {"name": "LAST_NAME"},
    {"name": "PERSON_NAME"},
    {"name": "FIRST_NAME"},
    {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "EMAIL_ADDRESS"},
    {"name": "STREET_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "US_TOLLFREE_PHONE_NUMBER"},
    {"name": "VEHICLE_IDENTIFICATION_NUMBER"}
]

def get_dlp_client(credentials_path):
    return dlp_v2.DlpServiceClient.from_service_account_file(credentials_path)

def get_dlp_project_id(credentials_path):
    project_id = ""
    with open(credentials_path, "r") as f:
            content = f.read()
            try:
                json_data = json.loads(content)
                project_id = json_data["project_id"]
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON content.")
    return project_id

def get_filtered_info_types(dlp_client):
    info_types = dlp_client.list_info_types(request={"location_id": "us", "location_id": "us"}).info_types
    # return [{"name": it.name} for it in info_types if it.name not in EXCLUDED_INFO_TYPES] 
    return SELECTED_INFO_TYPES

def get_inspect_config(info_types):
    inspect_config = {
    "info_types": info_types,  # Already a list of {"name": ...}
    "include_quote": False,
    "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
    }
    return inspect_config

def split_text_chunks(text, max_bytes=50_000):
    chunks = []
    current = []
    total = 0
    for line in text.splitlines():
        line_size = len(line.encode("utf-8"))
        if total + line_size > max_bytes:
            chunks.append("\n".join(current))
            current = [line]
            total = line_size
        else:
            current.append(line)
            total += line_size
    if current:
        chunks.append("\n".join(current))
    return chunks


def deidentify_text(dlp_client, project_id, text, info_types, inspect_config, file_path):
    print("Inside function deidentify_text")
    deid_config = {
        "info_type_transformations": {
            "transformations": [{
                "info_types": info_types,
                "primitive_transformation": {
                    "replace_with_info_type_config": {}
                }
            }]
        }
    }

    if file_path.endswith(".json") or file_path.endswith(".jsonl"):
        try:
            json_data = json.loads(text)
            chunks = split_text_chunks(json_data["text"])
            deid_chunks = [
                deidentify_content(dlp_client, project_id, deid_config, inspect_config, {"value": chunk})
                for chunk in chunks
            ]
            json_data["text"] = "\n".join(deid_chunks) 
            # item = {"value":json_data["text"]}
            # json_data["text"] = deidentify_content(dlp_client, project_id, deid_config, inspect_config, item)
            return json.dumps(json_data)            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON content.")

    else:
        # #item = {"value": text}
        # # return deidentify_content(dlp_client, project_id, deid_config, item) 
        chunks = split_text_chunks(text)
        deid_chunks = [
            deidentify_content(dlp_client, project_id, deid_config, inspect_config, {"value": chunk})
            for chunk in chunks
        ]
        return "\n".join(deid_chunks)
        # dlp_client.deidentify_content(
        #     request={
        #         "parent": f"projects/{project_id}",
        #         "deidentify_config": deid_config,
        #         "item": item
        #     }
        # )
        # return response.item.value

def deidentify_content(dlp_client, project_id, deid_config, inspect_config, item):
    requestParams = {
        "parent": f"projects/{project_id}",
        "deidentify_config": deid_config,
        "inspect_config": inspect_config,
        "item": item
    }
    request = DeidentifyContentRequest(requestParams)
    response = dlp_client.deidentify_content(request=request
            # request={
            #     "parent": f"projects/{project_id}",
            #     "deidentify_config": deid_config,
            #     "item": item
            # }
        )
    return response.item.value

def async_deidentify(file_access_key, case_file_id, job_type, file_paths_dict, dlp_client, project_id, info_types, inspect_config, data):
    for original_path, deidentified_path in file_paths_dict.items():
        print(f"🔄 Start of de-identification for file: {original_path}")
        try:            
            original_content = read_file_from_firebase(original_path)
            deidentified_text = deidentify_text(
                dlp_client, project_id, original_content, info_types, inspect_config, original_path
            )
            save_deidentified_file_to_gcs(deidentified_path, deidentified_text)
            
            print(f"✅ End of de-identification for file: {original_path}")
        except Exception as e:
            print(f"❌ inside async exception handling: Error processing {original_path}: {str(e)}", flush=True)
            if not isinstance(e, DeIdentifierPipelineError):
                e = DeIdentifierPipelineError("Unhandled error in async_deidentify", context={
                    "stage": "async_deidentify",
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

    
