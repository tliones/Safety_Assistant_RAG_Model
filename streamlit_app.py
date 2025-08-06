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

def clean_latex_output(text):
    import re

    # Remove \text{} and \mathrm{} (causes rendering issues)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{(.*?)\}", r"\1", text)

    # Inline math: wrap common symbols in $
    inline_terms = [r"C_i", r"T_i", r"\\sum", r"\\times"]
    for term in inline_terms:
        text = re.sub(rf"(?<!\$)\b({term})\b(?!\$)", r'$\1$', text)

    # Block math: wrap \frac{...}{...} lines in $$...$$ if not already
    text = re.sub(r"(?<!\$)\s*(\\frac\{[^}]+\}\{[^}]+\})\s*(?!\$)", r"$$\1$$", text)

    # Remove any leftover double-prefix like "TWA = $$..."
    text = re.sub(r"^\s*\w+\s*=\s*\$\$(.*?)\$\$", r"$$\1$$", text, flags=re.MULTILINE)

    return text






# Load secrets securely from Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

st.title("Safety Document QA Assistant - Dropbox Version")

# Replace this with your actual Dropbox paths (inside App Folder if using App Folder permission)
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

def load_csv_from_dropbox(path):
    _, res = dbx.files_download(path)
    return pd.read_csv(BytesIO(res.content))

def load_npy_from_dropbox(path):
    _, res = dbx.files_download(path)
    return np.load(BytesIO(res.content), allow_pickle=True)

selected_docs = st.multiselect("Select document sources to search:", list(DOCUMENTS.keys()), default=[])

all_dfs = []
all_embeddings = []

for doc_name in selected_docs:
    paths = DOCUMENTS[doc_name]
    try:
        df = load_csv_from_dropbox(paths['csv'])
        embeddings = load_npy_from_dropbox(paths['npy'])
        all_dfs.append(df)
        all_embeddings.append(embeddings)
    except Exception as e:
        st.error(f"Error loading {doc_name}: {e}")

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

        prompt = f"Context:\n{full_context}\n\nQuestion: {question}\n\nInstructions: If your answer includes formulas, place them on their own line using LaTeX inside double dollar signs like $$...$$. Do not include math inside narrative sentences. Avoid using \\text{{}}. Only write formulas using raw LaTeX syntax, and never prefix them with words like 'TWA ='.\n\nAnswer:"



        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful safety assistant. When outputting formulas, use valid LaTeX inside double dollar signs ($$). Avoid using \\text{} — just write plain variables or use \\mathrm{} if needed."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        st.session_state.answer = response.choices[0].message.content
        st.session_state.minimal_context = minimal_context
        st.session_state.full_context = full_context

    if st.session_state.answer:
        st.subheader("Answer:")

        from streamlit.components.v1 import html
        def split_and_render(answer):
            # Split into lines
            lines = answer.split("\n")
            for line in lines:
                # Check if it's a LaTeX block (e.g., matches \frac or similar)
                if re.match(r"^\s*\\(frac|sum|int|begin|end|cdot|times|text|sqrt|alpha|beta)", line.strip()):
                    st.latex(line.strip())
                else:
                    st.markdown(clean_latex_output(line), unsafe_allow_html=True)

                    
        split_and_render(clean_latex_output(st.session_state.answer))

    
        st.subheader("Sources:")
        st.write(st.session_state.minimal_context)

        if st.button("Show Full Context"):
            st.subheader("Full Context:")
            st.write(st.session_state.full_context)

else:
    st.warning("No documents selected or no valid files found.")

