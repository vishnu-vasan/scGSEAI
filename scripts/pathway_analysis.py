import argparse
import os
import warnings
import pandas as pd
import gseapy as gp
from typing import Union
import time
# from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

def get_env_variable(var_name: str) -> str:
    """
    Retrieve an environment variable or raise a detailed error.
    """
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"[ERROR] Environment variable '{var_name}' is not set.")
    return value

def initialize_llm(model: str) -> Union[Gemini, Groq]:
    """
    Initialize and return the appropriate LLM based on the model name.
    
    Args:
        model (str): The name of the model (e.g., "gemini-pro", "llama2-70b").

    Returns:
        An instance of either Google Gemini or Groq model.
    """
    if model.startswith("gemini"):
        api_key = get_env_variable("GEMINI_API_KEY")
        print(f"INFO: Using Gemini model: {model}")
        return Gemini(model=f"models/{model}", api_key=api_key)
    else:
        api_key = get_env_variable("GROQ_API_KEY")
        print(f"INFO: Using Groq model: {model}")
        return Groq(model=model)

def detect_input_files(input_dir):
    """Automatically detect all gene expression files (.xlsx or .csv) and a context text file from the input directory."""
    gene_expression_files = []
    context_file = None

    for file in os.listdir(input_dir):
        if file.endswith((".xlsx", ".csv")):
            gene_expression_files.append(os.path.join(input_dir, file))
        elif file.endswith(".txt") and not context_file:
            context_file = os.path.join(input_dir, file)

    if not gene_expression_files:
        raise FileNotFoundError("Error: No gene expression files found starting with 'de_results_' in the input directory.")

    return gene_expression_files, context_file

def process_gene_expression(file_path, top_genes):
    """Load and filter gene expression data from an Excel file."""
    print("[INFO] Loading gene expression data...")
    # Read based on file extension
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")
    
    # Select relevant columns
    # Flexible column mapping
    col_mapping = {
        'gene': ['Gene', 'gene', 'names', 'genes'],
        'avg_log2FC': ['avg_log2FC', 'logfoldchanges', 'log2FC'],
        'p_val_adj': ['p_val_adj', 'pvals_adj', 'p_vals_adj']
    }

    mapped_cols = {}
    for key, options in col_mapping.items():
        for col in options:
            if col in df.columns:
                mapped_cols[key] = col
                break
        if key not in mapped_cols:
            raise ValueError(f"Missing required column for '{key}'. Acceptable options: {options}")

    # Select relevant columns
    selected_cols = [mapped_cols['gene'], mapped_cols['avg_log2FC'], mapped_cols['p_val_adj']]
    df_selected = df[selected_cols]

    # check here later on for avg_log2FC column name
    df_selected.columns = ['gene', 'avg_log2FC', 'p_val_adj']  # Rename for internal consistency
    
    # Filter for statistical significance (p_val_adj < 0.05)
    df_significant = df_selected[df_selected['p_val_adj'] < 0.05]
    
    # Separate into upregulated and downregulated genes
    df_upregulated = df_significant[df_significant['avg_log2FC'] > 0]
    df_downregulated = df_significant[df_significant['avg_log2FC'] < 0]

    print(f"[INFO] Found {len(df_upregulated)} upregulated genes and {len(df_downregulated)} downregulated genes.")
    
    # Sort by adjusted p-value first (ascending), then by absolute fold change (descending)
    df_upregulated_sorted = df_upregulated.sort_values(by=['p_val_adj', 'avg_log2FC'], ascending=[True, False]).head(top_genes)
    df_downregulated_sorted = df_downregulated.sort_values(by=['p_val_adj', 'avg_log2FC'], ascending=[True, True]).head(top_genes)
    
    print(f"[INFO] Found {len(df_upregulated_sorted)} upregulated genes and {len(df_downregulated_sorted)} downregulated genes after sorting and extracting top {top_genes}.")

    return df_upregulated_sorted, df_downregulated_sorted

def run_enrichr_analysis(gene_list, organism, regulation_type):
    """Perform pathway enrichment analysis using Enrichr."""
    if (not gene_list.empty):
        print(f"[INFO] Running Enrichr analysis for {regulation_type} genes...")
        enr = gp.enrichr(
            gene_list=gene_list['gene'].tolist(),
            gene_sets=['GO_Biological_Process_2023'],
            organism=organism,
            outdir=None  # Do not write to disk
        )
        
        enr_df = enr.results

        for col in ['Overlap', 'Old P-value', 'Old Adjusted P-value']:
            if col in enr_df.columns:
                enr_df.drop(columns=[col], inplace=True)

        # Sort the dataframe by Adjusted P-value
        enr_df.sort_values(by='Adjusted P-value', ascending=True, inplace=True)
        # filter for non significant adj p-vals

        print(f"[INFO] Found {len(enr_df)} significantly enriched pathways for {regulation_type} genes.")
        time.sleep(5)

        return enr_df
    else:
        print(f"No genes in: {regulation_type} category")
        return

def dataframe_to_text(df, regulation_type):
    """Convert a DataFrame of enriched pathways into structured text."""
    text_data = f"=== {regulation_type} Pathways ===\n\n"
    for _, row in df.iterrows():
        text_data += (
            f"- **Pathway:** {row['Term']} ({row['Gene_set']})\n"
            f"  - **P-value:** {row['P-value']}\n"
            f"  - **Adjusted P-value:** {row['Adjusted P-value']}\n"
            f"  - **Odds Ratio:** {row['Odds Ratio']}\n"
            f"  - **Combined Score:** {row['Combined Score']}\n"
            f"  - **Genes:** {row['Genes']}\n\n"
        )
    return text_data.strip()

def perform_rag_query(query):
    """Perform a query using the indexed Enrichr results."""
    global query_engine
    if query_engine is None:
        return "[ERROR] Please index Enrichr results first."

    print("[INFO] Performing LLM query on indexed data...")
    try:
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"[ERROR] Query processing failed: {str(e)}"

def index_enrichr_results(text_data, llm, embed_model):
    """Index the Enrichr results for RAG-based querying."""
    global index, query_engine
    print("[INFO] Indexing enriched pathways for RAG-based querying...")

    try:
        if not text_data:
            return "[ERROR] No data provided for indexing."

        # Convert to LlamaIndex Document
        document = Document(text=text_data)

        # Create index in-memory
        index = VectorStoreIndex.from_documents([document], llm=llm, embed_model=embed_model)

        # Create query engine
        query_engine = index.as_query_engine()

        return "[INFO] Enrichr results successfully indexed."
    except Exception as e:
        return f"[ERROR] Indexing failed: {str(e)}"

def load_biological_context(file_path):
    """Load biological context from a text file."""
    print("[INFO] Loading biological context...")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def main(args):
    
    model_name = args.llm
    llm = initialize_llm(model_name)

    # Detect input files
    gene_expression_files, context_file = detect_input_files(args.input_dir)

    # Load biological context once
    if context_file:
        biological_context = load_biological_context(context_file)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    for expression_file in gene_expression_files:
        cell_type = os.path.splitext(os.path.basename(expression_file))[0].replace("de_results_", "")

        print(f"\n[INFO] Processing cell type: {cell_type}")

        try:
            # Load gene expression data
            df_upregulated, df_downregulated = process_gene_expression(expression_file, top_genes=args.top_genes)

            # Run Enrichr analysis
            up_regulated_enr_df = run_enrichr_analysis(df_upregulated, args.organism, "Upregulated")
            down_regulated_enr_df = run_enrichr_analysis(df_downregulated, args.organism, "Downregulated")

            # Skip if both are None
            if up_regulated_enr_df is None and down_regulated_enr_df is None:
                print(f"[WARNING] No significant enrichment found for {cell_type}. Skipping...")
                continue

            # Create output directory for enrichr results if it doesn't exist
            os.makedirs(args.enrichr_dir, exist_ok=True)

            # Save Enrichr Results

            if up_regulated_enr_df is not None:
                up_regulated_enr_df.to_csv(f"{args.enrichr_dir}/{cell_type}_Upregulated_KO_Up_Enrichr_Results.csv", index=False)
                print(f"[INFO] Saved Enrichr Results for Upregulated genes for {cell_type}.")
            else:
                print(f"[INFO] No Enrichr results for Upregulated genes for {cell_type}.")

            if down_regulated_enr_df is not None:
                down_regulated_enr_df.to_csv(f"{args.enrichr_dir}/{cell_type}_Downregulated_KO_Down_Enrichr_Results.csv", index=False)
                print(f"[INFO] Saved Enrichr Results for Downregulated genes for {cell_type}.")
            else:
                print(f"[INFO] No Enrichr results for Downregulated genes for {cell_type}.")

            # Convert DataFrames to structured text
            upregulated_text = dataframe_to_text(up_regulated_enr_df, "KO-Up")
            downregulated_text = dataframe_to_text(down_regulated_enr_df, "KO-Down")
            combined_text = f"{upregulated_text}\n\n{downregulated_text}"

            if not combined_text:
                print(f"[INFO] No pathway data to index for {cell_type}. Skipping RAG query.")
                continue

            # # Set up LlamaIndex
            print("[INFO] Setting up LLM and embedding model...")
            Settings.llm = llm
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # # Index data for RAG
            indexing_status = index_enrichr_results(combined_text, llm, embed_model)
            print(indexing_status)

            # Define structured query (unchanged)
            query = f'''
                This is my context for a single-cell RNA-seq study focusing on a specific cell type within colon tissues:
                <start of context>
                The orphan nuclear receptor Nr4a1 (Nur77) has been implicated in regulating apoptosis, immune responses, and metabolic processes. Its possible dysregulation of Nr4a1 can be particularly relevant in cancer, where its improper function may lead to uncontrolled cell proliferation and defective apoptosis. Additionally, emerging evidence suggests Nr4a1 plays a role in aging and dietary responses, further underscoring its importance in maintaining homeostasis. To investigate the whole-body impact of Nr4a1 loss, we conducted a single-cell RNA sequencing (scRNA-seq) analysis of colon tissues from wild-type (WT) and Nr4a1 knockout (KO) mice.
                
                Our premise is that Nr4a1 is a 'bad guy' in the context of cancer, despite using healthy samples for this study. We are comparing gene expression in a specific cell type within the colon tissue of healthy Nr4a1 knockout (KO) mice versus healthy wild-type (WT) mice.
                
                The study employs **single-cell RNA sequencing (scRNA-seq)** to analyze differential gene expression within a specific **{cell_type}** cell population across these conditions.
                
                Enrichment analysis has already been performed externally via Enrichr using top {args.top_genes} differentially expressed genes for both conditions.
                
                <task>
                You have access to embedded pathway enrichment results from Enrichr for both KO-up and KO-down gene sets.

                1. **Identify Enriched Terms**:
                    - List **only those** enriched biological terms with their adjusted p-values (e.g., pathways, Gene Ontology terms, etc.) associated with Nr4a1 knockout (KO) {cell_type} cells that are **statistically significant with adjusted p-values strictly less than 0.05**.
                    - List **only those** enriched biological terms with their adjusted p-values (e.g., pathways, Gene Ontology terms, etc.) associated with wild-type (WT) {cell_type} cells that are **statistically significant with adjusted p-values strictly less than 0.05**.
                    - **Exclude all terms with adjusted p-values ≥ 0.05**. If no terms meet the threshold, return "None found".
                    
                2. **Biological Insights**:
                    - Discuss the potential functions or processes that are activated or suppressed due to Nr4a1 loss.
                    - Highlight any implications for cancer biology, even in the healthy tissue context.
                    - Assess whether any of the enriched terms suggest potential therapeutic targets or novel biological mechanisms relevant to Nr4a1 and its role in cellular function, potentially connecting to cancer.
                
                3. **Recurrent Genes in Enriched Terms**:
                    - Identify the most **frequently occurring genes** across the statistically significant enriched terms listed in Section 1 for both KO-up and KO-down sets.
                
                4. **Summarized Output**:

                    - **Top Enriched Terms (Refined Selection)**:  
                        - From the **statistically significant enriched terms** identified in Section 1, present a **refined list** of top enriched biological terms for both KO-up and KO-down groups.  
                        - The selection must consider **multiple criteria simultaneously**:
                            - **Lower adjusted p-values** (greater statistical significance),
                            - **Larger number of associated genes per term** (indicating broader pathway involvement),
                            - **Reduction in redundancy**, especially among Gene Ontology (GO) terms (e.g., using semantic similarity or clustering heuristics to avoid listing overlapping terms).
                        - The goal is to present a **concise, representative set** of enriched terms that **maximize biological informativeness** for each group.

                    - **Biological Implications Summary**:  
                        - Summarize the potential biological implications of these enriched terms in relation to the study's premise about Nr4a1.
                </task>
                '''

            # Perform RAG-based query
            response = perform_rag_query(query)

            # Save response
            output_file_name = f"{cell_type}_pathway_analysis_response.txt"
            output_file_path = os.path.join(args.output_dir, output_file_name)

            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(response)

            print(f"[INFO] Analysis for {cell_type} saved to: {output_file_path}")
        
        except Exception as e:
            print(f"[ERROR] Failed to process {cell_type}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Pathway Analysis with Enrichr & RAG")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files (Excel & context text)")
    parser.add_argument("--enrichr_dir", type=str, required=True, help="Directory containing enrichr files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output results")
    parser.add_argument("--llm", type=str, required=True, help="Which LLM to use", default="deepseek-r1-distill-qwen-32b")
    parser.add_argument("--top_genes", type=int, required=True, help="Number of top genes to select")
    parser.add_argument("--organism", type=str, default="human", help="Organism (default: human)")

    args = parser.parse_args()
    main(args)