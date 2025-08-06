import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import dropbox
import os
from io import BytesIO
import re

# --- Load secrets ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

st.title("Safety Document QA Assistant - Dropbox Version")

# --- Dropbox file paths ---
DOCUMENTS = {
    "Company Safety Manual": {
        "csv": "/company_safety_manual_sections.csv",
        "npy": "/company_safety_manual_embeddings.npy"
    },
    "OSHA 1910 Standards": {
        "csv": "/osha_1910_standards_sections.csv",
        "npy": "/osha_1910_standards_embeddings.npy"
    },
    "NIOSH Heat Stress Guide": {
        "csv": "/niosh_heat_stress_guide_sections.csv",
        "npy": "/niosh_heat_stress_guide_embeddings.npy"
    }
}

# --- Dropbox file loading ---
def load_csv_from_dropbox(path):
    _, res = dbx.files_download(path)
    return pd.read_csv(BytesIO(res.content))

def load_npy_from_dropbox(path):
    _, res = dbx.files_download(path)
    return np.load(BytesIO(res.content), allow_pickle=True)

# --- Clean and render LaTeX and markdown ---
def clean_and_render_response(text):
    # Clean up common LaTeX issues
    text = re.sub(r'\\mug', r'\\mu g', text)
    text = re.sub(r'µg', r'\\mu g', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Handle empty lines
        if not line:
            st.write("")
            i += 1
            continue
        
        # Handle block LaTeX ($$...$$)
        if line.startswith('$$') and line.endswith('$$') and len(line) > 4:
            formula = line[2:-2].strip()
            try:
                st.latex(formula)
            except Exception as e:
                st.code(f"LaTeX Error: {formula}")
            i += 1
            continue
        
        # Handle multiline block LaTeX
        if line.startswith('$$') and not line.endswith('$$'):
            formula_lines = [line[2:]]  # Remove opening $$
            i += 1
            
            # Collect lines until we find closing $$
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.endswith('$$'):
                    formula_lines.append(next_line[:-2])  # Remove closing $$
                    i += 1
                    break
                else:
                    formula_lines.append(next_line)
                    i += 1
            
            formula = '\n'.join(formula_lines).strip()
            try:
                st.latex(formula)
            except Exception as e:
                st.code(f"LaTeX Error: {formula}")
            continue
        
        # Process line for inline LaTeX and markdown
        processed_line = process_inline_latex(line)
        
        try:
            st.markdown(processed_line, unsafe_allow_html=True)
        except Exception as e:
            st.write(line)
        
        i += 1

def process_inline_latex(text):
    """Process inline LaTeX in markdown text"""
    # Find all inline LaTeX expressions ($...$)
    def replace_inline_latex(match):
        latex_content = match.group(1)
        # Clean up the LaTeX content
        latex_content = latex_content.strip()
        return f"${latex_content}$"
    
    # Replace inline LaTeX expressions
    processed = re.sub(r'\$([^$]+)\$', replace_inline_latex, text)
    return processed

# --- UI for document selection ---
selected_docs = st.multiselect("Select document sources to search:", list(DOCUMENTS.keys()), default=[])

all_dfs = []
all_embeddings = []

for doc_name in selected_docs:
    try:
        df = load_csv_from_dropbox(DOCUMENTS[doc_name]["csv"])
        emb = load_npy_from_dropbox(DOCUMENTS[doc_name]["npy"])
        all_dfs.append(df)
        all_embeddings.append(emb)
    except Exception as e:
        st.error(f"Error loading {doc_name}: {e}")

# --- Main logic ---
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_embeddings = np.vstack(all_embeddings)

    model = SentenceTransformer("all-mpnet-base-v2")

    question = st.text_input("Ask your safety question:")

    if 'answer' not in st.session_state:
        st.session_state.answer = ""
        st.session_state.minimal_context = ""
        st.session_state.full_context = ""

    if st.button("Get Answer") and question:
        query_vec = model.encode([question])
        similarities = cosine_similarity(query_vec, combined_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:3]
        top_matches = combined_df.iloc[top_indices]

        minimal_context = ""
        full_context = ""

        for _, row in top_matches.iterrows():
            section_info = f"Source: {row.get('source', 'Unknown')}"
            if not pd.isnull(row.get('section_number')):
                section_info += f" - Section {row['section_number']}"
            if not pd.isnull(row.get('section_title')):
                section_info += f": {row['section_title']}"
            if not pd.isnull(row.get('page')):
                section_info += f" (Page {int(row['page'])})"

            minimal_context += f"{section_info}\n"
            full_context += f"{section_info}\n{row['text']}\n\n"

        # RAG Prompt - Updated for better LaTeX formatting
        prompt = f"""Context:
{full_context}

Question: {question}

Instructions: 
1. For mathematical formulas that should be displayed as blocks, place each formula on its own line surrounded by double dollar signs: $$formula$$
2. For inline math variables or simple expressions, use single dollar signs: $x$
3. Do NOT use \\text{{}} or \\mathrm{{}} commands - use plain text instead
4. For units, write them as plain text: "5 mg/m³" instead of "5 \\text{{mg/m³}}"
5. Use standard markdown formatting (bullets, bold, italics) as needed
6. Keep LaTeX simple and avoid complex formatting commands

Answer:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful safety assistant. Use simple LaTeX notation for math: $$....$$ for block formulas and $...$ for inline math. Avoid complex LaTeX commands. Write units as plain text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            st.session_state.answer = response.choices[0].message.content
            st.session_state.minimal_context = minimal_context
            st.session_state.full_context = full_context

        except Exception as e:
            st.error(f"Error getting response from OpenAI: {e}")

    # --- Render output ---
    if st.session_state.answer:
        st.subheader("Answer:")
        clean_and_render_response(st.session_state.answer)

        st.subheader("Sources:")
        st.write(st.session_state.minimal_context)

        if st.button("Show Full Context"):
            st.subheader("Full Context:")
            st.write(st.session_state.full_context)

else:
    st.warning("No documents selected or no valid files found.")


