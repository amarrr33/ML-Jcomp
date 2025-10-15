#!/usr/bin/env python3
"""
Sample Viewer - Shows examples from the expanded dataset
"""

import os

import csv
import json

def show_samples():
    # Use organized paths
    data_file = 'data/cyberseceval3-visual-prompt-injection-expanded.csv'
    source_file = 'data/cyberseceval3-visual-prompt-injection-expanded_sources.json'
    
    if not os.path.exists(data_file):
        print(f"❌ Dataset not found at: {data_file}")
        print("Run: python3 scripts/expand_dataset.py first")
        return
    
    print("\n" + "="*80)
    print("CYBERSECEVAL3 EXPANDED DATASET - SAMPLE VIEWER")
    print("="*80 + "\n")
    
    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    with open(source_file, 'r') as f:
        sources = json.load(f)
    
    # Create source lookup
    source_map = {s['id']: s for s in sources}
    
    # Show original English example
    print("📝 SAMPLE 1: Original English Entry")
    print("-" * 80)
    entry = data[0]
    source = source_map.get(entry['id'], {})
    print(f"ID: {entry['id']}")
    print(f"Source: {source.get('source', 'N/A')} - {source.get('source_details', 'N/A')}")
    print(f"Language: {source.get('language', 'N/A')}")
    print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
    print(f"User Input: {entry['user_input_text']}")
    print(f"Image Description: {entry['image_description']}")
    print(f"Image Text: {entry['image_text'][:80]}...")
    print(f"Injection Technique: {entry['injection_technique']}")
    print(f"Risk Category: {entry['risk_category']}")
    
    # Show French translation
    print("\n\n🇫🇷 SAMPLE 2: French Translation")
    print("-" * 80)
    entry = data[1000]
    source = source_map.get(entry['id'], {})
    print(f"ID: {entry['id']}")
    print(f"Source: {source.get('source', 'N/A')} - {source.get('source_details', 'N/A')}")
    print(f"Language: {source.get('language', 'N/A')}")
    print(f"Original ID: {source.get('original_id', 'N/A')}")
    print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
    print(f"User Input: {entry['user_input_text']}")
    print(f"Image Description: {entry['image_description']}")
    print(f"Injection Technique: {entry['injection_technique']}")
    print(f"Risk Category: {entry['risk_category']}")
    
    # Show Russian translation
    print("\n\n🇷🇺 SAMPLE 3: Russian Translation")
    print("-" * 80)
    entry = data[1020]
    source = source_map.get(entry['id'], {})
    print(f"ID: {entry['id']}")
    print(f"Source: {source.get('source', 'N/A')} - {source.get('source_details', 'N/A')}")
    print(f"Language: {source.get('language', 'N/A')}")
    print(f"Original ID: {source.get('original_id', 'N/A')}")
    print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
    print(f"User Input: {entry['user_input_text']}")
    print(f"Image Description: {entry['image_description']}")
    print(f"Injection Technique: {entry['injection_technique']}")
    print(f"Risk Category: {entry['risk_category']}")
    
    # Show Tamil translation
    print("\n\n🇮🇳 SAMPLE 4: Tamil Translation")
    print("-" * 80)
    entry = data[1040]
    source = source_map.get(entry['id'], {})
    print(f"ID: {entry['id']}")
    print(f"Source: {source.get('source', 'N/A')} - {source.get('source_details', 'N/A')}")
    print(f"Language: {source.get('language', 'N/A')}")
    print(f"Original ID: {source.get('original_id', 'N/A')}")
    print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
    print(f"User Input: {entry['user_input_text']}")
    print(f"Image Description: {entry['image_description']}")
    print(f"Injection Technique: {entry['injection_technique']}")
    print(f"Risk Category: {entry['risk_category']}")
    
    # Show Hindi translation
    print("\n\n🇮🇳 SAMPLE 5: Hindi Translation")
    print("-" * 80)
    entry = data[1060]
    source = source_map.get(entry['id'], {})
    print(f"ID: {entry['id']}")
    print(f"Source: {source.get('source', 'N/A')} - {source.get('source_details', 'N/A')}")
    print(f"Language: {source.get('language', 'N/A')}")
    print(f"Original ID: {source.get('original_id', 'N/A')}")
    print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
    print(f"User Input: {entry['user_input_text']}")
    print(f"Image Description: {entry['image_description']}")
    print(f"Injection Technique: {entry['injection_technique']}")
    print(f"Risk Category: {entry['risk_category']}")
    
    # Show synthesized example
    print("\n\n⚡ SAMPLE 6: Synthesized English Example")
    print("-" * 80)
    entry = data[1080]
    source = source_map.get(entry['id'], {})
    print(f"ID: {entry['id']}")
    print(f"Source: {source.get('source', 'N/A')} - {source.get('source_details', 'N/A')}")
    print(f"Language: {source.get('language', 'N/A')}")
    print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
    print(f"User Input: {entry['user_input_text']}")
    print(f"Image Description: {entry['image_description']}")
    print(f"Image Text: {entry['image_text'][:80]}...")
    print(f"Injection Technique: {entry['injection_technique']}")
    print(f"Risk Category: {entry['risk_category']}")
    
    # Statistics
    print("\n\n📊 DATASET STATISTICS")
    print("-" * 80)
    
    # Count by language
    lang_counts = {}
    for s in sources:
        lang = s.get('language', 'unknown')
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print("\nEntries by Language:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang.capitalize():15} {count:4} entries ({count/len(data)*100:5.1f}%)")
    
    # Count by source type
    source_counts = {}
    for s in sources:
        src = s.get('source', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print("\nEntries by Source Type:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src.replace('_', ' ').title():25} {count:4} entries ({count/len(data)*100:5.1f}%)")
    
    # Count by risk category
    risk_counts = {}
    for entry in data:
        risk = entry.get('risk_category', 'unknown')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print("\nEntries by Risk Category:")
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk.replace('-', ' ').title():25} {count:4} entries ({count/len(data)*100:5.1f}%)")
    
    # Count by injection type
    inj_counts = {}
    for entry in data:
        inj = entry.get('injection_type', 'unknown')
        inj_counts[inj] = inj_counts.get(inj, 0) + 1
    
    print("\nEntries by Injection Type:")
    for inj, count in sorted(inj_counts.items()):
        print(f"  {inj.capitalize():25} {count:4} entries ({count/len(data)*100:5.1f}%)")
    
    print("\n" + "="*80)
    print(f"TOTAL ENTRIES: {len(data)}")
    print("="*80 + "\n")

if __name__ == '__main__':
    show_samples()
