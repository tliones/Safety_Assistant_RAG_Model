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
    
    # Split into paragraphs instead of lines for better processing
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        
        if not paragraph:
            continue
            
        lines = paragraph.split('\n')
        
        # Check if this is a standalone formula (contains LaTeX symbols and math operators)
        is_formula_block = False
        if len(lines) == 1:
            line = lines[0].strip()
            # Detect formulas that should be rendered as LaTeX blocks
            if (re.search(r'[=+\-*/]', line) and 
                re.search(r'\\[a-zA-Z]+|_\{[^}]+\}|\^\{[^}]+\}|[_{^]', line) and
                not line.startswith('#') and not line.startswith('*') and not line.startswith('-')):
                is_formula_block = True
        
        if is_formula_block:
            formula = lines[0].strip()
            # Remove any existing $ wrapping
            if formula.startswith('$') and formula.endswith('$'):
                formula = formula[2:-2]
            try:
                st.latex(formula)
            except Exception as e:
                st.code(f"Formula: {formula}")
        else:
            # Process as markdown with potential inline LaTeX
            for line in lines:
                line = line.strip()
                if not line:
                    st.write("")
                    continue
                
                # Handle explicit block LaTeX ($...$)
                if line.startswith('$') and line.endswith('$') and len(line) > 4:
                    formula = line[2:-2].strip()
                    try:
                        st.latex(formula)
                    except Exception as e:
                        st.code(f"Formula: {formula}")
                    continue
                
                # Process as regular markdown with inline LaTeX support
                try:
                    st.markdown(line, unsafe_allow_html=True)
                except Exception as e:
                    st.write(line)

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

    model = SentenceTransformer("all-mpnet-base-v2", device = "cpu")

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
1. For mathematical formulas, write them WITHOUT dollar signs or LaTeX wrappers - just the plain formula
2. Put important formulas on their own lines
3. For variables in text, you can use standard notation like T_wb, T_g, etc.
4. Use regular text formatting and bullet points
5. Write units as plain text: "5 mg/m³" or "degrees C"
6. Keep formatting simple and clear

Answer:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful safety assistant. Write mathematical formulas as plain text without dollar signs. The app will automatically detect and render them. Use clear formatting with formulas on separate lines when appropriate."},
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
