# 🔬 Automated Pathway Analysis Pipeline

This repository provides a streamlined pipeline for:
- **Processing differential gene expression (DGE) data**  
- **Performing pathway enrichment analysis using Enrichr**  
- **Retrieval-Augmented Generation (RAG) with LlamaIndex**  
- **Context-aware pathway interpretation using an LLM (Groq API)**  

---

## 📁 Directory Structure

```
pathway-analysis/
│── input/                    # Input directory (Excel gene expression & context text file)
│── output/                   # Output directory (Enrichr & LLM results)
│── scripts/                  # Main Python scripts
│   ├── pathway_analysis.py   # Main script for enrichment & LLM analysis
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
```

---

## ⚙️ **Setup Instructions**

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/vishnu-vasan/scGSEAI
cd scGSEAI
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Set up the Environment Variable**
Before running the script, export your Groq API key:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```
Here is a short video on how to create a Groq API Key: [Groq API Key Setup](https://youtu.be/qbUELF9Et4s?si=5uMuxESOuqjaBiS4)

---

## 🚀 **Running the Pipeline**

### **1️⃣ Prepare Input Files**
- **Gene Expression Data:** A `.xlsx` file containing `gene`, `avg_log2FC`, and `p_val_adj` columns.
- **Biological Context:** A `.txt` file describing the experiment/study background.

### **2️⃣ Run the Python Script**
```bash
python scripts/pathway_analysis.py --input/ input --output output/ --organism human
```

### **3️⃣ View Output**
- Enrichment results will be saved in **`output/`**.
- LLM-generated pathway analysis in **`pathway_analysis_response.txt`**.

---

## Upcoming - 📓 **Tutorial Notebook**
A Jupyter Notebook (`notebooks/pathway_analysis_tutorial.ipynb`) will be included in the near-future for step-by-step guidance.

To run it:
```bash
jupyter notebook notebooks/pathway_analysis_tutorial.ipynb
```

---

## 📜 **License**
This project is licensed under the **MIT License**.

---

## 🔗 **References**
- [GSEApy](https://gseapy.readthedocs.io/)
- [Enrichr API](https://maayanlab.cloud/Enrichr/)
- [LlamaIndex](https://gpt-index.readthedocs.io/)
- [Groq LLM](https://console.groq.com/)

## **Contributors**
- Dr James J Cai
- Romero Gonzalez, Selim S
- Shreyan Gupta
- Vishnuvasan Raghuraman