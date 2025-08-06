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
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            st.write("")
            continue

        # Fix LaTeX issues
        line = re.sub(r'\\mug', r'\\mu\\text{g}', line)
        line = re.sub(r'µg', r'\\mu\\text{g}', line)
        line = re.sub(r'\\text\{([^}]*)\}', r'\1', line)  # remove \text{}
        line = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', line)

        # Full block LaTeX
        if line.startswith('$$') and line.endswith('$$'):
            try:
                st.latex(line[2:-2])
            except Exception:
                st.markdown(f"`{line}`")
            continue

        # Inline-only LaTeX line
        if re.fullmatch(r'\$[^$]+\$', line):
            try:
                st.latex(line[1:-1])
            except Exception:
                st.markdown(f"`{line}`")
            continue

        # General mixed markdown (with inline $...$)
        try:
            st.markdown(line, unsafe_allow_html=True)
        except Exception:
            st.markdown(f"`{line}`")

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

        # RAG Prompt
        prompt = f"""Context:
{full_context}

Question: {question}

Instructions: 
1. If your answer includes mathematical formulas, place each formula on its own line surrounded by double dollar signs: $$formula$$
2. Use inline math like $x$ for variables.
3. Do NOT use \\text{{}} or \\mathrm{{}} commands.
4. Keep formatting simple and compatible with LaTeX parsers.
5. Use bullet points or markdown if applicable.

Answer:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful safety assistant. Respond with markdown text and clean LaTeX inside $$...$$ or $...$. Do not use unnecessary formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
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


