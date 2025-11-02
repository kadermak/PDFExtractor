# PDFExtractor
PDF Extractor

README

Date: 01112025

#### USAGE ####
# 1. in main function, Set 'GOOGLE_API_KEY' , currently set to default environment variable.
# 2. Ensure PDF is accessible via ' pdf_url or local_pdf_path ' path. If URL is not reachable will use local copy
# 3. Run the script in a notebook or console. For task3, enter user query in input box

#### Overview ####

This project automates the extraction and analysis of financial and date-related information from the Singapore Budget 2024 PDF.
Using a combination of Python, PyMuPDF, regular expressions, and the Google Gemini 2.5 generative model via LangChain, it performs:

# Tasks:
#   ✅ Task 1: Structured Data Extraction
#   ✅ Task 2: Date Normalization & Reasoning
#   ✅ Task 3: Multi-Agent Supervisor (Revenue + Expenditure)
#   ✅ Interactive Chat + Save Chat Log


#### DATA FLOW ####

[Budget PDF] 
      │
      ▼
[PyMuPDF Text Extraction] ──► [Regex Preprocessing: Locate Key Rows]
      │                                    │
      ▼                                    ▼
[Cleaned Text with Markers] ─────────────► [LLM Prompted Extraction of Figures]
      │                                    │
      ▼                                    ▼
[Validation & Correction] ───────────────► [Date Extraction & Status Reasoning]
      │                                    │
      ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│               Data Structuring & JSON Output                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        [Multi-Agent Query Supervisor w/ UI]


#### PYTHON PACKAGE ####

---

## Key Python Packages

| Package               | Purpose                      | Description                            |
|-----------------------|------------------------------|--------------------------------------|
| PyMuPDF (`fitz`)      | PDF parsing                  | Fast, efficient text extraction      |
| `re`                  | Regex matching              | Flexible pattern extraction          |
| `requests`            | HTTP fetching               | Download PDFs securely                |
| `langchain_core` & `langchain_google_genai` | LLM orchestration          | Prompt chaining and model calls      |
| `json`                | Data parsing/formatting     | Standardized structured output       |
| `datetime`            | Date handling               | Parsing and logical date comparison  |
| `ipywidgets`          | Interactive UI in notebooks | User-friendly input/output UI         |

---
 

#### Function-Level Summary #####
 
| Function Name           | Purpose                                                                                       | Description                                                                                                      |
|------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `__init__`             | Initialize extractor                                                                           | Sets API key, PDF URLs/paths, pages to extract, and initializes the LLM model instance.                         |
| `fetch_and_extract_text`| Download and extract PDF text                                                                 | Downloads PDF from web or local, extracts text from specified pages with PyMuPDF, returns combined raw text.     |
| `preprocess_cit_row`    | Locate and clean Corporate Income Tax row                                                    | Uses regex to identify and mark "Corporate Income Tax" row for LLM parsing, handles irregular formatting.        |
| `clean_text_for_llm`    | Clean text for LLM consumption                                                               | Removes currency symbols and normalizes whitespace from text to improve model input quality.                     |
| `adjust_fiscal_position`| Correct fiscal position sign errors                                                          | Detects and fixes common misreads in fiscal data sign, ensuring official positive value for FY2022 surplus.      |
| `extract_financial_data`| Extract structured numeric financial data                                                    | Coordinates preprocessing and prompts LLM to extract numeric finance data as JSON. Prints and returns results.   |
| `extract_dates_and_reason`| Extract and interpret budget-related dates                                                  | Uses prompts to extract key dates, then classifies them as expired/upcoming/ongoing relative to FY2024.           |
| `revenue_agent`         | Revenue-focused query handling                                                               | Routes revenue-related queries to LLM with relevant context to generate detailed budget revenue answers.         |
| `expenditure_agent`     | Expenditure-focused query handling                                                           | Routes spending and fund allocation queries to LLM with contextual budget text for detailed responses.           |
| `supervisor_agent`      | Query router and multi-agent coordinator                                                    | Routes queries to revenue or expenditure agents or general summarizer based on keywords, tracks conversation.    |
| `run_interactive_ui`    | Jupyter Notebook interactive question UI                                                    | Provides ipywidgets-based UI for live questioning, displaying routed responses, and saving chat logs.            |
| `run_console_loop`      | CLI interactive question loop                                                                | Console-based interface for asking budget questions, printing routed answers, and saving logs on exit.           |

---

 


High-Level Design
flowchart TD
    A[PDF Document] --> B[PDF Extraction (PyMuPDF)]
    B --> C[Pre-processing: Clean CIT row (Regex)]
    C --> D[Task 1: Finance JSON Extraction (Gemini 2.5)]
    C --> E[Task 2: Date Extraction & Reasoning (Gemini 2.5 + Python)]
    C --> F[Task 3: Multi-Agent Supervisor]
    F --> G1[Revenue Agent]
    F --> G2[Expenditure Agent]
    F --> G3[Supervisor]
    G1 --> H[Query Response]
    G2 --> H
    G3 --> H

[Budget PDF Document]
         │
         ▼
 [PyMuPDF Text Extraction]
         │
         ▼
  [Regex Preprocessing]
         │
         ▼
 [Cleaned Text with Markers]
         │
         ▼
  [LLM Prompted Numeric Extraction]
         │
         ▼
 [Validation & Correction] 
         │
         ▼
  [Date Extraction & Reasoning]
         │
         ▼
       [JSON Output]
         │
         ▼
 [Multi-Agent Query Interface]

#### Design Considerations #####
#### Task 1: Structured Finance JSON

Goal: Extract corporate income tax, top-ups, revenue taxes, and fiscal position.

Approach:

Robust extraction of Corporate Income Tax row using regex.

Preprocess text into pipe-separated columns for structured parsing.

Use Gemini 2.5 to convert natural text to JSON.

Benefits:

Handles line breaks, inconsistent spacing, and negative numbers.

Converts millions to billions automatically.

JSON output ensures easy downstream processing.

#### Task 2: Date Extraction + Reasoning

Goal: Extract and normalize important fiscal dates.

Approach:

Gemini 2.5 extracts document_distribution_date and estate_duty_date.

Python logic classifies dates relative to 2024-01-01.

Benefits:

Normalized date format (YYYY-MM-DD) for consistent reporting.

Provides automated reasoning (Expired / Upcoming / Ongoing).

### Task 3: Multi-Agent Supervisor

Goal: Enable ad-hoc Q&A on budget document.

Approach:

Supervisor agent routes queries to Revenue or Expenditure agent based on keywords.

Each agent uses Gemini 2.5 with a contextual prompt.

Benefits:

Modular design allows easy addition of new agents.

Provides detailed, focused answers to user queries.

Chat log stored for auditing and analysis.

Package Selection
Package	Purpose
requests	Download PDF from web URL
fitz (PyMuPDF)	Extract text from PDF pages
re	Regex-based robust pre-processing
io.BytesIO	Handle in-memory PDF content
datetime	Normalize and reason about dates
langchain_core	Prompt templates, chaining, and JSON parsing
langchain_google_genai	Gemini 2.5 model integration
ipywidgets	Notebook-based interactive UI
json	Structured output and chat log storage
Function Summary
Function	Description
extract_relevant_text(url, pages_indices)	Fetches PDF from URL/local file and extracts relevant pages as text.
preprocess_cit_row(text)	Locates and cleans the Corporate Income Tax row for structured extraction.
classify_date(normalized_date_str)	Compares a date to 2024-01-01 and labels it as Expired, Upcoming, or Ongoing.
revenue_agent(query, context)	Handles revenue-focused queries in Task 3.
expenditure_agent(query, context)	Handles expenditure-focused queries in Task 3.
supervisor_agent(query, context)	Routes query to the correct agent based on keywords.
LLM chains (RunnableSequence)	Converts text to structured JSON and extracts dates.



### Benefits of Design

Robustness: Handles irregular PDF formatting and negative values.

Modularity: Each task is isolated with clear input/output.

Scalability: Additional agents or fields can be added easily.

Auditability: JSON outputs and chat logs provide full traceability.

Interactive: Supports Jupyter UI for rich user interaction.

##### Key Learnings & Error Handling
1. Handling PDF Variability and Corruption
PDF layouts vary widely; text extraction might fail if PDFs are corrupted or contain scanned images instead of embedded text.

PyMuPDF is robust and fast but may miss text or return garbled output if the PDF uses image text or unusual fonts.

Recommendation: Detect and handle missing text gracefully; fallback to OCR if necessary.

Recovery: Tools like mutool or pre-processing with PyMuPDF can clean corrupt files.

2. Regex Preprocessing Resilience
Text extracted from PDFs may include irregular line breaks, spacing, or hidden characters.

Regex patterns must account for optional line breaks and multiple spaces to reliably identify target rows (e.g., Corporate Income Tax).

Tip: Use multiline regex with fallback strategies to improve row detection.

3. LLM Prompt Engineering & Validation
Explicit and clear prompts improve extraction accuracy from noisy or semi-structured text.

Unit conversion instructions (millions to billions) and explicit sign rules (parentheses = negative) reduce errors.

Always validate and correct LLM outputs with custom post-processing to catch anomalies (e.g., wrongly signed fiscal positions).

4. Graceful Exception Handling
Network failures in downloading PDFs require fallback mechanisms (local copies or retries).

PDF parsing exceptions should be caught to prevent total pipeline failure.

Validate all intermediate outputs and include logging or print statements for debugging extraction issues.

5. Interactive Query Routing
Implementing specialized agents based on query content improves response relevance and efficiency.

UI elements like ipywidgets add usability but require fallback for non-notebook environments.

Additional Error Handling Techniques in PyMuPDF
Catch RuntimeError and similar exceptions during PDF open or text extraction.

Validate extracted text length and patterns; if empty or suspicious, trigger alternative methods.

################

 
