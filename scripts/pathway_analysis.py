import argparse
import os
import warnings
import pandas as pd
import gseapy as gp
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


def detect_input_files(input_dir):
    """Automatically detect the Excel file and context text file from the input directory."""
    excel_file = None
    context_file = None

    for file in os.listdir(input_dir):
        if file.endswith(".xlsx") and not excel_file:
            excel_file = os.path.join(input_dir, file)
        elif file.endswith(".txt") and not context_file:
            context_file = os.path.join(input_dir, file)

    if not excel_file:
        raise FileNotFoundError("Error: No Excel (.xlsx) file found in the input directory.")
    if not context_file:
        raise FileNotFoundError("Error: No context (.txt) file found in the input directory.")

    return excel_file, context_file


def process_gene_expression(file_path):
    """Load and filter gene expression data from an Excel file."""
    print("[INFO] Loading gene expression data...")
    df = pd.read_excel(file_path)
    
    # Select relevant columns
    selected_cols = ['gene', 'avg_log2FC', 'p_val_adj']
    df_selected = df[selected_cols]
    
    # Filter for statistical significance (p_val_adj < 0.05)
    df_significant = df_selected[df_selected['p_val_adj'] < 0.05]
    
    # Separate into upregulated and downregulated genes
    df_upregulated = df_significant[df_significant['avg_log2FC'] > 0]
    df_downregulated = df_significant[df_significant['avg_log2FC'] < 0]

    print(f"[INFO] Found {len(df_upregulated)} upregulated genes and {len(df_downregulated)} downregulated genes.")
    
    return df_upregulated, df_downregulated


def run_enrichr_analysis(gene_list, organism, regulation_type):
    """Perform pathway enrichment analysis using Enrichr."""
    print(f"[INFO] Running Enrichr analysis for {regulation_type} genes...")
    
    enr = gp.enrichr(
        gene_list=gene_list['gene'].tolist(),
        gene_sets=['GO_Biological_Process_2023'],
        organism=organism,
        outdir=None  # Do not write to disk
    )
    
    # Filter results by Adjusted P-value < 0.05
    enr_df = enr.results[enr.results['Adjusted P-value'] < 0.05]
    enr_df.drop(columns=['Overlap', 'Old P-value', 'Old Adjusted P-value'], inplace=True)

    print(f"[INFO] Found {len(enr_df)} significantly enriched pathways for {regulation_type} genes.")
    
    return enr_df


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
    # Check if GROQ_API_KEY is set
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("[ERROR] GROQ_API_KEY is not set. Please export it as an environment variable.")

    # Detect input files
    excel_file, context_file = detect_input_files(args.input_dir)

    # Load gene expression data
    df_upregulated, df_downregulated = process_gene_expression(excel_file)

    # Run Enrichr analysis
    up_regulated_enr_df = run_enrichr_analysis(df_upregulated, args.organism, "Upregulated")
    down_regulated_enr_df = run_enrichr_analysis(df_downregulated, args.organism, "Downregulated")

    # Convert DataFrames to structured text
    upregulated_text = dataframe_to_text(up_regulated_enr_df, "Upregulated")
    downregulated_text = dataframe_to_text(down_regulated_enr_df, "Downregulated")
    combined_text = f"{upregulated_text}\n\n{downregulated_text}"

    # Set up LlamaIndex
    print("[INFO] Setting up LLM and embedding model...")
    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = Groq(model="deepseek-r1-distill-qwen-32b")
    Settings.llm = llm
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Index data for RAG
    indexing_status = index_enrichr_results(combined_text, llm, embed_model)
    print(indexing_status)

    # Load biological context from file
    biological_context = load_biological_context(context_file)

    # Define structured query (unchanged)
    query = f'''
        This is my context for a single-cell RNA-seq study:
        <start of context>
        {biological_context}
        <end of context>

        <task>
        1. **Identify Relevant Pathways**:
        - Based on the upregulated (disease-associated) and downregulated (healthy/WT) genes, determine key biological pathways enriched in the experimental conditions.
        - Highlight shifts in pathway activity across experimental groups, considering the study's biological focus.

        2. **Stepwise Pathway Analysis**:
        - **Step 1**: Identify pathways enriched in the disease condition and their functional roles.
        - **Step 2**: Determine pathways that are suppressed in disease but enriched in healthy conditions.
        - **Step 3**: Compare how a therapeutic intervention (if applicable) modulates pathway activation.

        3. **Biological Insights**:
        - Provide a structured interpretation of the molecular mechanisms driving the observed phenotype.
        - Assess whether any pathway shifts suggest potential therapeutic targets or novel biological mechanisms.
        - Highlight cell-type specificity in pathway activation (if applicable).

        4. **Summarized Output**:
        - Present findings in a structured format.
        - Ensure clarity, highlighting major takeaways relevant to the study's objectives.
        </task>
    '''

    # Perform RAG-based query
    response = perform_rag_query(query)

    # Save response
    output_file_path = os.path.join(args.output_dir, "pathway_analysis_response.txt")
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(response)

    print(f"[INFO] Analysis complete. Results saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Pathway Analysis with Enrichr & RAG")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files (Excel & context text)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output results")
    parser.add_argument("--organism", type=str, default="human", help="Organism (default: human)")

    args = parser.parse_args()
    main(args)
