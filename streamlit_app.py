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

# Load secrets securely from Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

st.title("Safety Document QA Assistant - Dropbox Version")

# Dropbox file paths
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

def clean_and_render_response(text):
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            st.write("")
            continue
        if (line.startswith('$$') and line.endswith('$$')) or \
           re.search(r'\\(frac|sum|int|sqrt|alpha|beta|gamma|delta|times|cdot|partial)', line):
            latex_line = line
            if latex_line.startswith('$$') and latex_line.endswith('$$'):
                latex_line = latex_line[2:-2]
            latex_line = re.sub(r'\\text\{([^}]*)\}', r'\1', latex_line)
            latex_line = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', latex_line)
            latex_line = re.sub(r'\b(mg|kg|g|lb|oz|ppm|ppb|°C|°F|K)\b', r'\\text{\1}', latex_line)
            latex_line = re.sub(r'\b(m|cm|mm|ft|in|yd)\b(?=[\^/])', r'\\text{\1}', latex_line)
            latex_line = re.sub(r'(\d+)\s*(mg|kg|g|lb|oz|ppm|ppb)', r'\1\\,\\text{\2}', latex_line)
            latex_line = re.sub(r'(mg|kg|g|lb|oz)/\s*(m|cm|mm|ft|in)', r'\\text{\1}/\\text{\2}', latex_line)
            latex_line = re.sub(r'^[A-Za-z\s]*=\s*', '', latex_line)
            latex_line = re.sub(r'^[A-Za-z\s]*:\s*', '', latex_line)
            try:
                st.latex(latex_line)
            except Exception:
                st.markdown(f"`{line}`")
        else:
            processed_line = re.sub(r'\$(.+?)\$', r'\\(\1\\)', line)
            processed_line = re.sub(r'\\text\{([^}]*)\}', r'\1', processed_line)
            st.markdown(processed_line, unsafe_allow_html=True)

# Main app
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

        prompt = f"""Context:
{full_context}

Question: {question}

Instructions: 
1. If your answer includes mathematical formulas, place each formula on its own line surrounded by double dollar signs: $$formula$$
2. Do NOT use \\text{{}} or \\mathrm{{}} commands
3. Do NOT prefix formulas with variable names or equals signs
4. Use standard LaTeX syntax for mathematical expressions
5. For inline math in sentences, use single dollar signs: $variable$
6. Keep formulas simple and avoid complex formatting

Answer:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful safety assistant. When outputting mathematical formulas, use clean LaTeX syntax inside double dollar signs ($$formula$$). Avoid \\text{} commands and prefixes. Place each formula on its own line."},
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

