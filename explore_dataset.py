import json
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset

def explore_msmarco_hi():
    print("Loading ai4bharat/MSMARCO-XI Hindi parquet file directly...")
    try:
        ds = load_dataset('parquet', data_files='hf://datasets/ai4bharat/MSMARCO-XI/train/hintrain.parquet', split="train", streaming=True)
        print("Dataset stream loaded.\n")
        
        # Take the first record
        record = next(iter(ds))
        
        print("--- SCHEMA INSPECTION ---")
        for key in record.keys():
            val = record[key]
            type_name = type(val).__name__
            if type_name == 'dict':
                print(f"- {key} ({type_name}): keys = {list(val.keys())}")
            elif type_name == 'list':
                print(f"- {key} ({type_name}): length = {len(val)}, first_elem_type = {type(val[0]) if len(val) > 0 else 'N/A'}")
            else:
                print(f"- {key} ({type_name})")
                
        print("\n--- RECORD DATA (SAMPLE) ---")
        print(f"Query (hi): {record.get('query')}")
        print(f"Eng_Query: {record.get('Eng_Query')}")
        print(f"Answer (hi): {record.get('Answer')}")
        print(f"Eng_Answer: {record.get('Eng_Answer')}")
        print(f"is_selected: {record.get('passages', {}).get('is_selected')}")
        
        passages = record.get('passages', {})
        english_passages = passages.get('English_passages', [])
        translated_passages = passages.get('Translated_passages', [])
        
        print(f"Number of English passages: {len(english_passages)}")
        print(f"Number of Translated passages: {len(translated_passages)}")
        print(f"First English passage snippet: {english_passages[0][:100]}..." if english_passages else "No English passages.")
        print(f"First Translated passage snippet: {translated_passages[0][:100]}..." if translated_passages else "No translated passages.")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    explore_msmarco_hi()
