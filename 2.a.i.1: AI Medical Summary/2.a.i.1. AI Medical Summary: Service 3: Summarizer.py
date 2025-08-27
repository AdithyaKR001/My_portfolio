import os
import json
import copy
import uuid
from PyPDF2 import PdfReader
from docx import Document
import openai
from openai import OpenAI
from textwrap import wrap
import httpx
import re
from google import genai
from google.oauth2 import service_account
from google.genai import types
from datetime import datetime
from .firebase_utils import read_file_from_firebase
import pandas as pd

class SummarizerPipelineError(Exception):
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

MAX_CHUNK_SIZE = 80_000  # ~3k tokens per chunk
MAX_SUMMARY_CHUNK_SIZE = 100_000  # safe input size for aggregation prompt
MORTALITY_TABLE_FOLDER_PATH = os.getenv("MORTALITY_TABLE_FOLDER_PATH", "/app/mortality_tables")

def read_api_key_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config.get("api_key")

def read_prompt_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_file(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif path.lower().endswith(".json") or path.endswith(".jsonl"):
        with open(path, "r") as f:
            return json.dumps(json.load(f).get("text")) #, indent=2)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def split_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    return wrap(text, max_chunk_size, break_long_words=False, replace_whitespace=False)

def summarize_chunk(chunk, prompt, gptClient, gptName):
    print(f"Summarizing using {gptName}", flush=True)
    model = ""
    if gptName == "chatgpt":
        print(f"Model name set to gpt-5-mini", flush=True)
        model = "gpt-5-mini"
        response = gptClient.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{prompt}\n\nInput:\n{chunk}"}
        ],
        stream = False
        )
    # print(response.choices[0].message.content)
        return response.choices[0].message.content
    elif gptName == "grok":
        print(f"Model name set to grok-3-latest", flush=True)
        model = "grok-3"
        response = gptClient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\nInput:\n{chunk}"}
                ],
                stream = False,
                temperature=0
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    elif gptName == "gemini":
        print(f"Model name set to gemini-2.5-flash", flush=True)
        model = "gemini-2.5-flash"
        contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"{prompt}\n\nInput:\n{chunk}")
                        ]
                    )
                ]


        response = gptClient.models.generate_content(
                    model=model,
                    contents= contents
                )
        return response.text
    
    # response = gptClient.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": f"{prompt}\n\nInput:\n{chunk}"}
    #     ],
    #     stream = False #,
    #     #temperature=0
    # )
    # # print(response.choices[0].message.content)
    # return response.choices[0].message.content

# def flatten_and_prefix_json(data, prefix="chatgpt", sep="."):
#     def _flatten(obj, parent_key=""):
#         items = {}
#         for k, v in obj.items():
#             new_key = f"{parent_key}{sep}{k}" if parent_key else k
#             if isinstance(v, dict):
#                 items.update(_flatten(v, new_key))
#             elif isinstance(v, list):
#                 items[f"{prefix}_{new_key}"] = "\\n".join(map(str, v))
#             else:
#                 items[f"{prefix}_{new_key}"] = str(v)
#         return items

#     return _flatten(data)
def convert_to_array(text):
    return [
        s.strip().rstrip('.').strip() + '.' 
        for s in text.split('\n') if s.strip()
    ]

def flatten_and_prefix_json(data, prefix="chatgpt", sep=".", delimiter="\n"):
    if prefix == "chatgpt":
        prefix = "c"
    elif prefix == "grok":
        prefix = "g"
    elif prefix == "gemini":
        prefix = "ge"

    # Fields that should be returned as a list of cleaned sentences
    sentence_array_keys = {
        "c_tobacco_and_substance_use",
        "g_tobacco_and_substance_use",
        "ge_tobacco_and_substance_use"
    }

    # Fields to zip into *_mortality_underwriting
    zip_keys = ["accelerated_mortality_factors", "underwriting_drivers"]

    def parse_sentence_array(text):
        # Split on newline or punctuation-based sentence boundaries
        raw_parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        result = []
        for s in raw_parts:
            s = s.strip()
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            result.append(s)
        return result

    def insert_newlines_after_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return delimiter.join([s.strip() for s in sentences if s.strip()])

    def try_parse_json_array(value):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return None

    def _flatten(obj, parent_key=""):
        items = {}
        local_store = {}

        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            full_key = f"{prefix}_{new_key}"

            if any(k.endswith(z) for z in zip_keys):
                local_store[k] = v
                continue

            if isinstance(v, dict):
                items.update(_flatten(v, new_key))

            elif isinstance(v, list):
                if all(isinstance(i, str) for i in v):
                    processed_list = [insert_newlines_after_sentences(item) for item in v]
                    items[full_key] = delimiter.join(processed_list)
                else:
                    items[full_key] = v  # keep list of objects as-is

            else:
                str_val = str(v).strip()

                # Handle tobacco/substance use fields as sentence array
                if full_key in sentence_array_keys:
                    items[full_key] = parse_sentence_array(str_val)

                # Handle stringified arrays
                elif str_val.startswith("[") and str_val.endswith("]"):
                    parsed_array = try_parse_json_array(str_val)
                    if parsed_array is not None:
                        items[full_key] = parsed_array
                    else:
                        items[full_key] = insert_newlines_after_sentences(str_val)

                else:
                    items[full_key] = insert_newlines_after_sentences(str_val)

        # Handle zipped "factor" + "driver" lists → *_mortality_underwriting
        for base_prefix in ["c", "g", "ge"]:
            if prefix != base_prefix:
                continue

            factor_key = f"{base_prefix}_accelerated_mortality_factors"
            driver_key = f"{base_prefix}_underwriting_drivers"
            output_key = f"{base_prefix}_mortality_underwriting"

            factor_raw = local_store.get("accelerated_mortality_factors")
            driver_raw = local_store.get("underwriting_drivers")

            if isinstance(factor_raw, list) and isinstance(driver_raw, list):
                zipped_result = []
                for i in range(max(len(factor_raw), len(driver_raw))):
                    zipped_result.append({
                        "factor": factor_raw[i]["factor"] if i < len(factor_raw) and "factor" in factor_raw[i] else "",
                        "driver": driver_raw[i]["driver"] if i < len(driver_raw) and "driver" in driver_raw[i] else ""
                    })
                items[output_key] = zipped_result

        return items

    return _flatten(data)

def shorten_flattened_keys(flat_dict, rename_map):
    result = {}
    for key, value in flat_dict.items():
        for long_suffix, short_suffix in rename_map.items():
            if key.endswith(long_suffix):
                key = key.replace(long_suffix, short_suffix)
                break
        result[key] = value
    return result


# Function to deeply merge JSON objects
def deep_merge(*dicts):
    def merge(a, b):
        for key in b:
            if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key])
            else:
                a[key] = copy.deepcopy(b[key])
        return a

    result = {}
    for d in dicts:
        merge(result, d)
    return result

def read_mortality_tables(mortality_dir):
    mortality_data_strs = []
    for filename in os.listdir(mortality_dir):
        if filename.lower().endswith(".csv"):
            category_name = os.path.splitext(filename)[0].strip()

            # Read CSV
            df = pd.read_csv(os.path.join(mortality_dir, filename))

            if df.shape[1] < 2:
                raise ValueError(f"CSV file {filename} does not have the expected 2 columns.")

            # Convert table to string without using iterrows (faster, no warnings)
            table_str = "\n".join(
                f"{int(age)}: {float(months)} months"
                for age, months in zip(df.iloc[:, 0], df.iloc[:, 1])
            )

            mortality_data_strs.append(f"Mortality Table - {category_name}:\n{table_str}")
    
    return "\n\n".join(mortality_data_strs)

def process_files_with_llms(paths, prompt_path, API_KEY_PATHS, job_type, summary_file_path, lshid):
    # openai.api_key = read_api_key_from_json(CHATGPT_API_KEY_PATH)
    TOTAL_PAGE_COUNT = 0
    timeout = httpx.Timeout(60.0, connect=600.0, read=600.0, write=600.0)
    # gptClient = OpenAI(api_key=read_api_key_from_json(CHATGPT_API_KEY_PATH), timeout=timeout)
    
    
    # Load prompts
    base_prompt = read_prompt_from_docx(prompt_path)

    if job_type == "aiMedicalSummary":
        mortality_tables_str = read_mortality_tables(MORTALITY_TABLE_FOLDER_PATH)
        # Append mortality data to base prompt
        base_prompt = (
            base_prompt
            + "\n\nThe following mortality tables are provided for reference:\n\n"
            + mortality_tables_str
        )

    if summary_file_path != "":
        # Add AI medical summary json file contents to the prompt for LE Arbitrage Report generation.
        summary_content = read_file_from_firebase(summary_file_path)
        additional_prompt_content = (
        "\n\n The following is the input Category A: AI-Generated Life Expectancy & Medical Summary json. Consider only the properties and their values and ignore the json opening and closing braces.\n"
        + f"{summary_content}"
        )
        base_prompt = base_prompt + additional_prompt_content

    chunk_prompt = (
        base_prompt
        + "\n\nSummarize the following chunk. Extract key elements like dates, diagnoses, treatments, outcomes, and lab/imaging findings."
    )
    global_prompt = (
        base_prompt +
        "\n\nGiven the following summaries from document chunks, generate a comprehensive summary that merges them into a final output. "
        "Maintain clarity, avoid repetition, and ensure all critical findings are included."
    )

    output_dir = os.path.dirname(paths[0])
    output_files = []
    llm_summaries = []
    # Combine content from all files
    combined_input = ""
    for path in paths:
        print(f"Extracting text from file: {path}", flush=True)
        content = read_file_from_firebase(path) #extract_text_from_file(path)
        combined_input += f"\n\n--- START OF {os.path.basename(path)} ---\n{content}\n--- END OF {os.path.basename(path)} ---"
        json_data = json.loads(content)
        TOTAL_PAGE_COUNT += int(json_data.get("pdf-total-pages", 0))

    # Step 1: Chunking
    chunks = split_text(combined_input)
    for key, filepath in API_KEY_PATHS.items():
        if key =="chatgpt" and job_type != "aiLEArbitrage":
            print("Instantiating ChatGPT instance....")
            gptClient = OpenAI(api_key=read_api_key_from_json(filepath), timeout=timeout)
        elif key == "grok":
            print("Instantiating Grok instance....", flush=True)
            gptClient =  OpenAI(api_key=read_api_key_from_json(filepath), base_url="https://api.x.ai/v1", timeout=timeout)
        elif key == "gemini" and job_type != "aiLEArbitrage":
            print("Instantiating Gemini instance....", flush=True)
            REQUIRED_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
            credentials = service_account.Credentials.from_service_account_file(
                filepath,
                scopes=REQUIRED_SCOPES
            )
            gptClient = genai.Client(
                vertexai=True,
                project="ls-hub-v2",
                location="global",
                credentials=credentials
            )            
        # Step 2: Local summarization
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}", flush=True)
            summary = summarize_chunk(chunk, chunk_prompt, gptClient, key)
            chunk_summaries.append(f"--- Chunk {i + 1} Summary ---\n{summary}")

        # Step 3: Global aggregation
        all_summaries_text = "\n\n".join(chunk_summaries)
        if len(all_summaries_text) > MAX_SUMMARY_CHUNK_SIZE:
            print(f"All summary text too large: {MAX_SUMMARY_CHUNK_SIZE}", flush=True)
            recursive_iteration = 0
            while len(all_summaries_text)>MAX_SUMMARY_CHUNK_SIZE:
                recursive_iteration += 1
                print(f"Recursive iteration: {recursive_iteration}", flush=True)
                current_summary_size = len(all_summaries_text)
                print(f"Current summary size: {current_summary_size}", flush=True)
                chunks = split_text(all_summaries_text) #, max_chunk_size=MAX_SUMMARY_CHUNK_SIZE)
                chunk_summaries = []
                
                for i, chunk in enumerate(chunks):
                    print(f"Summarizing chunk {i + 1}/{len(chunks)}", flush=True)
                    summary = summarize_chunk(chunk, chunk_prompt, gptClient, key)
                    chunk_summaries.append(f"--- Chunk {i + 1} Summary ---\n{summary}")
                all_summaries_text = "\n\n".join(chunk_summaries)
                
            # final_summary_parts = []
            # for part in summary_chunks:
            #     final_summary_parts.append(summarize_chunk(part, global_prompt, gptClient))
            # final_output = "\n\n".join(final_summary_parts)
        # else:
        # print(f"Current summary size before final summarization: {len(all_summaries_text)}")
        print("All summary text not too large", flush=True)
        print(f"Final output summarization using {key}", flush=True)
        final_output = summarize_chunk(all_summaries_text, global_prompt, gptClient, key)

        # Save to unique file
        # output_dir = os.path.dirname(paths[0])
        # output_file = os.path.join(output_dir, f"{key}_medical_summary_{uuid.uuid4().hex}.json")
        # with open(output_file, "w") as f:
        if final_output is not None and len(final_output) > 0:
            cleaned_output = (
                                final_output
                                .replace("```json", "")
                                .replace("```", "")
                                .replace("\n", "")
                                .strip()
            )
            json_obj = json.loads(cleaned_output)
            # if job_type != "aiLEArbitrage":
            json_obj = flatten_and_prefix_json(json_obj, key, "_")
            
            if "c_tobacco_and_substance_use" in json_obj:
                json_obj["c_tobacco_and_substance_use"] = convert_to_array(json_obj["c_tobacco_and_substance_use"])
            
            if "g_tobacco_and_substance_use" in json_obj:
                json_obj["g_tobacco_and_substance_use"] = convert_to_array(json_obj["g_tobacco_and_substance_use"])
            
            if "ge_tobacco_and_substance_use" in json_obj:
                json_obj["ge_tobacco_and_substance_use"] = convert_to_array(json_obj["ge_tobacco_and_substance_use"])
            

            rename_map = {
                        "patient_information_gender": "pi_gender",
                        "patient_information_date_of_birth": "pi_dob",
                        "patient_information_age": "pi_age",
                        "patient_information_ethnicity": "pi_ethnicity",
                        "summary_metrics_life_expectancy": "sm_life_expectancy",
                        "summary_metrics_comorbidities":"sm_comorbidities",
                        "summary_metrics_progression": "sm_progression",
                        "summary_metrics_compliance": "sm_compliance",
                        "summary_metrics_frailty": "sm_frailty",
                        "summary_metrics_mental_status": "sm_mental_status",
                        "summary_metrics_bmi_category": "sm_bmi_category",
                        "summary_metrics_tobacco_use_status": "sm_tobacco_use_status",
                        "summary_metrics_living_situation": "sm_living_situation",
                        "summary_comparison_ai_life_expectancy_months_c_sm_life_expectancy": "c_le",
                        "summary_comparison_ai_life_expectancy_months_g_sm_life_expectancy": "g_le",
                        "summary_comparison_ai_life_expectancy_months_ge_sm_life_expectancy": "ge_le",
                        "summary_comparison_market_life_expectancy_months": "comp_mark_le",
                        "summary_comparison_average_difference_months": "comp_avg_diff",
                        "summary_comparison_percentage_difference": "comp_per_diff",
                        "summary_comparison_arbitrage_opportunity": "comp_arb_opp",
                        "summary_comparison_confidence_score": "comp_conf_score",
                        "underwriting_comparison_conditions_overlooked_in_market": "und_comp_cond_ovr",
                        "underwriting_comparison_conditions_overweighted_in_market": "und_comp_cond_ovr_wt",
                        "underwriting_comparison_severity_discrepancies": "und_comp_sev_disc",
                        "underwriting_comparison_number_of_total_comorbidities_ai_summary": "und_comp_comorb_ai",
                        "underwriting_comparison_number_of_total_comorbidities_market_summary": "und_comp_comorb_mark",
                        "narrative_summary": "narr_summ"
                        }
            # json_obj["pi_age"] = json_obj.copy("c_patient_information_age")
            # json_obj["pi_gender"] = json_obj.get("c_patient_information_gender")
            json_obj = shorten_flattened_keys(json_obj, rename_map)
                

            # json.dump(json_obj, f, ensure_ascii=False, indent=2)
            llm_summaries.append(json_obj)
    
    # if job_type != "aiLEArbitrage": 
    json_obj_final = deep_merge(*llm_summaries)
    # else:
    #     json_obj_final = {"summary": llm_summaries}
    json_obj_final["Case_ID"] = lshid
    json_obj_final["total_page_count"] = TOTAL_PAGE_COUNT
    json_obj_final["date"] = datetime.now().strftime("%m/%d/%Y")
    print("LLM Summarization Complete.", flush=True)
    return json_obj_final

def generate_meta_summary(prompt_path, input_json, API_KEY_PATHS):
   prompt = read_prompt_from_docx(prompt_path)
   combined_text = build_combined_text(input_json)
   return get_summary_from_grok(prompt, combined_text, API_KEY_PATHS)

def build_combined_text(data):
    return (
        f"Summary 1: {data.get('g_health_summary_paragraph', '')}\n\n"
        f"Summary 2: {data.get('ge_health_summary_paragraph', '')}\n\n"
        f"Summary 3: {data.get('c_health_summary_paragraph', '')}"
    )

def get_summary_from_grok(prompt, text, API_KEY_PATHS):
    key = "grok"
    filepath = API_KEY_PATHS[key]        
    timeout = httpx.Timeout(60.0, connect=600.0, read=600.0, write=600.0)

    print("Instantiating Grok instance....", flush=True)
    gptClient =  OpenAI(api_key=read_api_key_from_json(filepath), base_url="https://api.x.ai/v1", timeout=timeout)
    print("Generating meta summary.", flush=True)
    return summarize_chunk(text, prompt, gptClient, key)