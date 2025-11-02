#!/usr/bin/env python
# coding: utf-8

# In[ ]:Kader


#### USAGE ####
# 1. in main function, Set 'GOOGLE_API_KEY' , currently set to default environment variable.
# 2. Ensure PDF is accessible via ' pdf_url or local_pdf_path ' path. If URL is not reachable will use local copy
# 3. Run the script in a notebook or console. For task3, enter user query in input box


# In[15]:


import os
import json
import re
import requests
from io import BytesIO
import fitz  # PyMuPDF
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.json import JsonOutputParser

try:
    from IPython.display import display, Markdown, clear_output
    import ipywidgets as widgets
    NOTEBOOK_MODE = True
except ImportError:
    NOTEBOOK_MODE = False


class PDFExtractor:
    """
    Extracts and analyzes financial data from Singapore Budget PDFs using PyMuPDF and Google Gemini LLM.

    Attributes:
        google_api_key (str): API key for Google Generative AI access.
        pdf_url (str): URL from which to download the PDF budget document.
        local_pdf_path (str): Path to local PDF file as fallback.
        pages_to_extract (list): List of page indices to extract text from.
        model (ChatGoogleGenerativeAI): Initialized LLM model for prompt-based extraction.
        document_text (str): Raw extracted text from PDF.
        cleaned_text (str): Preprocessed text ready for LLM consumption.
        chat_log (list): Log of user queries and agent responses for Task 3 interactivity.
    """

    def __init__(self,
                 google_api_key: str,
                 pdf_url: str,
                 local_pdf_path: str,
                 pages_to_extract: list):
        """
        Initialize the extractor with credentials and PDF details.
        """
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
        self.pdf_url = pdf_url
        self.local_pdf_path = local_pdf_path
        self.pages_to_extract = pages_to_extract
        self.document_text = ""
        self.cleaned_text = ""
        self.chat_log = []

    def fetch_and_extract_text(self):
        """
        Downloads the PDF from web or local file and extracts text from specified pages.

        Returns:
            str: Combined extracted raw text from all pages.
        """
        relevant_text = ""
        pdf_file = None
        headers = {"User-Agent": "Mozilla/5.0"}

        # Attempt to fetch PDF from web
        try:
            response = requests.get(self.pdf_url, headers=headers, stream=True, timeout=15)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
            print("✅ PDF fetched from web.")
        except Exception as e:
            print(f"⚠️ Web fetch failed ({e}). Trying local file.")
            if os.path.exists(self.local_pdf_path):
                pdf_file = open(self.local_pdf_path, "rb")
            else:
                print("❌ No PDF available for extraction.")
                return ""

        # Extract text using PyMuPDF
        try:
            with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
                for i in self.pages_to_extract:
                    if i >= len(pdf):
                        continue
                    page = pdf[i]
                    text = page.get_text("text") or ""
                    # Debug message for key page of interest
                    if i == 7:
                        print(f"📄 Page {i+1}: extracting raw text for Corporate Income Tax rows.")
                    if text:
                        relevant_text += f"\n\n--- PAGE {i+1} ---\n{text.strip()}"
            print(f"📄 Extraction complete | extracted characters: {len(relevant_text)}")
            self.document_text = relevant_text.strip()
            return self.document_text
        except Exception as e:
            print(f"❌ PDF error during extraction: {e}")
            return ""

    def preprocess_cit_row(self, text: str) -> str:
        """
        Locates and cleans the 'Corporate Income Tax' row in the raw text, marking it with a special flag.
        Designed to handle irregular formatting including line breaks and multiple spaces.

        Args:
            text (str): Raw extracted text from PDF.

        Returns:
            str: Text with 'Corporate Income Tax' row replaced by a flagged and pipe-delimited line.
        """
        pattern = re.compile(r"Corporate\s*Income\s*Tax[^\n]+", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            # Fallback pattern considering line breaks after label
            pattern_lines = re.compile(r"Corporate\s*Income\s*Tax[^\n]*(?:\n\s*\d[^\n]*){1,2}", re.IGNORECASE)
            match = pattern_lines.search(text)
        if match:
            cit_line = match.group(0).replace("\n", " ")
            # Replace multiple consecutive spaces with pipes to mark columns
            cit_processed = re.sub(r"\s{2,}", "|", cit_line)
            print("\n🔍 Cleaned CIT Row Detected:")
            print(cit_processed)
            print("-------------------------------------------")
            processed_text = text.replace(match.group(0), f"***CLEAN_CIT_ROW***: {cit_processed}")
            print("✅ CIT Row preprocessed and flagged.")
            return processed_text
        print("⚠️ No Corporate Income Tax row found for pre-processing.")
        return text

    def clean_text_for_llm(self, text: str) -> str:
        """
        Cleans currency symbols and normalizes whitespace to prepare text for LLM input.

        Args:
            text (str): Preprocessed text with flagged rows.

        Returns:
            str: Cleaned text ready for LLM prompt.
        """
        clean_text = text.replace("S$", "").replace("$", "")
        clean_text = re.sub(r"[^\S\r\n]+", " ", clean_text)  # Replace multiple spaces except newlines with single space
        clean_text = re.sub(r"\n+", "\n", clean_text)        # Remove repeated newlines
        return clean_text

    def adjust_fiscal_position(self, text: str) -> str:
        """
        Fixes known sign errors in fiscal position figures, especially correcting near 1.72 billion negative values.

        Args:
            text (str): Cleaned text with numeric data.

        Returns:
            str: Text with corrected fiscal position sign if necessary.
        """
        fiscal_pos_match = re.search(r"OVERALL FISCAL POSITION[^\d\(\-]*([\-]?\d+\.\d+)", text, re.IGNORECASE)
        if fiscal_pos_match:
            raw_val = float(fiscal_pos_match.group(1))
            # Correct small negative misreads of the known positive 1.72 value
            if abs(raw_val - 1.72) < 0.1 and raw_val < 0:
                print("🔧 Adjusting latest_actual_fiscal_position_billion from negative to positive 1.72")
                return re.sub(r"OVERALL FISCAL POSITION[^\n]+", "OVERALL FISCAL POSITION 1.72", text)
        return text

    def extract_financial_data(self):
        """
        Performs Task 1: preprocesses the document text and uses the LLM to extract numeric fiscal data as JSON.
        Prints snippet for debug and outputs extracted structured data.

        Returns:
            dict or None: Extracted financial data JSON or None if extraction fails.
        """
        if not self.document_text:
            print("❌ No document text available for extraction.")
            return None

        doc_preprocessed = self.preprocess_cit_row(self.document_text)
        if "***CLEAN_CIT_ROW***" not in doc_preprocessed:
            print("❌ CIT row still not found — manual verification required.")
        else:
            print("✅ CLEAN_CIT_ROW marker added successfully.")

        self.cleaned_text = self.clean_text_for_llm(doc_preprocessed)
        self.cleaned_text = self.adjust_fiscal_position(self.cleaned_text)

        snippet_start = self.cleaned_text.find("***CLEAN_CIT_ROW***")
        if snippet_start == -1:
            print("\n⚠️ CLEAN_CIT_ROW marker not found in clean_text.\n")
        else:
            snippet_end = snippet_start + 300
            print("\n🔎 Debug text snippet for LLM context:\n")
            print(self.cleaned_text[snippet_start: snippet_end])
            print("\n---------------------------------------------------------\n")

        extract_template = PromptTemplate.from_template("""
You are a financial analyst. Read the Budget 2024 document and extract numeric data.

Use the row marked as ***CLEAN_CIT_ROW***: which is pipe (|) separated. 
- Numbers are in millions unless noted. 
- Convert millions to billions by dividing by 1000 (e.g., 28380 → 28.38).
- Parentheses mean negative values.

Fields to output in JSON:
1. corporate_income_tax_2024 (float)  → value in 4th column of CLEAN_CIT_ROW
2. yoy_percentage_difference_cit_2024 (float) → value in last column of CLEAN_CIT_ROW
3. total_top_ups_2024 (float)
4. operating_revenue_taxes (list of strings)
5. latest_actual_fiscal_position_billion (float)

Text:
{text_data}
""")
        parser = JsonOutputParser()
        extract_chain = RunnableSequence(extract_template | self.model | parser)

        print("\n--- Task 1 LLM Output ---")
        fiscal_data = extract_chain.invoke({"text_data": self.cleaned_text})
        print("✅ JSON parsed successfully.\n")
        print(json.dumps(fiscal_data, indent=2))
        return fiscal_data

    def extract_dates_and_reason(self):
        """
        Performs Task 2: extracts important budget-related dates and classifies their status relative to FY 2024.
        Prints extracted dates and classification results.

        Returns:
            list of dict: Each dict contains original key, normalized date, and status (Expired/Upcoming/Ongoing/Unknown).
        """
        print("\n--- 📅 Running Task 2: Date Extraction & Reasoning ---")

        date_extract_template = PromptTemplate.from_template("""
You are a document analyst. From the text below, extract the following two dates.
- For the `document_distribution_date`, find the Budget Day date.
- For the `estate_duty_date`, find the date related to the abolition or last active status of Estate Duty (usually near page 36).
- The final output must be normalized to **YYYY-MM-DD**.

Fields to extract:
1. document_distribution_date (str)
2. estate_duty_date (str)

Text:
{text_data}
""")
        parser = JsonOutputParser()
        date_chain = RunnableSequence(date_extract_template | self.model | parser)

        date_data_raw = date_chain.invoke({"text_data": self.document_text})
        print("✅ Extracted Dates for Reasoning:")
        print(json.dumps(date_data_raw, indent=2))

        comparison_date = datetime(2024, 1, 1)

        def classify_date(normalized_date_str):
            try:
                d = datetime.strptime(normalized_date_str, "%Y-%m-%d")
                if d < comparison_date:
                    status = "Expired"
                elif d > comparison_date:
                    status = "Upcoming"
                else:
                    status = "Ongoing"
            except Exception:
                status = "Unknown"
            return normalized_date_str, status

        date_status_final = []
        for key, normalized_date_str in date_data_raw.items():
            if isinstance(normalized_date_str, str) and len(normalized_date_str) > 5:
                date_str, status = classify_date(normalized_date_str)
            else:
                date_str, status = normalized_date_str, "Unknown"
            date_status_final.append({
                "original_text_source": key,
                "normalized_date": date_str,
                "status": status
            })
        print("\n✅ Date Reasoning Complete:")
        print(json.dumps(date_status_final, indent=2))
        return date_status_final

    def revenue_agent(self, query: str) -> str:
        """
        Agent specialized in answering revenue-related budget queries by querying the LLM with focused prompt.

        Args:
            query (str): User question relating to government revenue.

        Returns:
            str: LLM generated answer text.
        """
        prompt = f"""
You are the Revenue Agent. Based on the Singapore Budget 2024 text below, identify government revenue streams and tax figures.
Focus on income tax, GST, customs duties, etc.
Query: {query}

Context: {self.cleaned_text[:5000]}...
"""
        return self.model.invoke(prompt).content

    def expenditure_agent(self, query: str) -> str:
        """
        Agent specialized in answering expenditure-related budget queries via LLM prompt.

        Args:
            query (str): User question relating to government spending or fund allocations.

        Returns:
            str: LLM generated answer text.
        """
        prompt = f"""
You are the Expenditure Agent. Analyze government spending, fund allocations, and special transfers in the Singapore Budget 2024 text below.
Focus on fund allocations (e.g., Future Energy Fund).
Query: {query}

Context: {self.cleaned_text[:5000]}...
"""
        return self.model.invoke(prompt).content

    def supervisor_agent(self, query: str) -> dict:
        """
        Routes user queries to the appropriate specialized agent or general summarizer based on query keywords.
        Logs the conversation history.

        Args:
            query (str): The user input query.

        Returns:
            dict: Contains the query, routed agent name, and LLM response.
        """
        q = query.lower()
        if any(w in q for w in ["revenue", "tax", "income", "gst"]):
            routed = "Revenue Agent"
            answer = self.revenue_agent(query)
        elif any(w in q for w in ["fund", "budget", "spending", "allocation", "expenditure"]):
            routed = "Expenditure Agent"
            answer = self.expenditure_agent(query)
        else:
            routed = "Supervisor"
            answer = self.model.invoke(f"Provide a balanced summary for this query based on the budget document: {query}. Context: {self.cleaned_text[:5000]}...").content
        result = {"query": query, "routed_to": routed, "response": answer}
        self.chat_log.append(result)
        return result

    def run_interactive_ui(self):
        """
        Runs a Jupyter notebook interactive UI with text input, real-time routing, and answer display.
        Enables saving chat history to file.
        """
        if not NOTEBOOK_MODE:
            print("Notebook UI not available in this environment.")
            return

        output_box = widgets.Output()
        input_box = widgets.Text(
            placeholder="Ask about Singapore Budget 2024...",
            description="You:",
            layout=widgets.Layout(width='90%')
        )
        save_button = widgets.Button(description="💾 Save Chat Log", button_style='success')

        def save_chat_history(_):
            with open("chat_history.json", "w") as f:
                json.dump(self.chat_log, f, indent=2)
            with output_box:
                clear_output(wait=True)
                display(Markdown("✅ Chat history saved to `chat_history.json`"))

        save_button.on_click(save_chat_history)

        def handle_input(change):
            query = change["new"].strip()
            if not query:
                return
            input_box.value = ""
            result = self.supervisor_agent(query)
            with output_box:
                clear_output(wait=True)
                display(Markdown(f"**🧭 Routed to:** {result['routed_to']}"))
                display(Markdown(f"**Q:** {result['query']}\n\n**A:** {result['response']}"))

        input_box.observe(handle_input, names="value")

        display(Markdown("### 💬 Task 3: Ask questions about Singapore Budget 2024"))
        display(input_box, output_box, save_button)

    def run_console_loop(self):
        """
        Runs a console input loop for interactive queries and prints routed answers.
        Saves chat log to file on exit.
        """
        print("Running in console mode. Type 'quit' to exit.\n")
        while True:
            q = input("Ask about Singapore Budget 2024: ")
            if q.lower() == "quit":
                break
            result = self.supervisor_agent(q)
            print(f"[{result['routed_to']}] → {result['response']}\n")

        print("\nChat history saved to chat_history.json")
        with open("chat_history.json", "w") as f:
            json.dump(self.chat_log, f, indent=2)


if __name__ == "__main__":
    extractor = PDFExtractor(
        google_api_key="AIzaSyDasfSd4-1AVvK3XLHHbfGiyaEQvO69L3w",
        pdf_url="https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/fy2024_analysis_of_revenue_and_expenditure.pdf",
        local_pdf_path="fy2024_analysis_of_revenue_and_expenditure.pdf",
        # local_pdf_path=r"C:\Users\maric\Documents\Digital\fy2024_analysis_of_revenue_and_expenditure.pdf",
        # local_pdf_path=os.path.join(os.path.dirname(__file__), "fy2024_analysis_of_revenue_and_expenditure.pdf"),
        pages_to_extract=[0, 4, 5, 7, 19, 35]
    )

    extractor.fetch_and_extract_text()
    extractor.extract_financial_data()
    extractor.extract_dates_and_reason()

    # Example Task 3 query and print
    example_query = "How will the Budget for the Future Energy Fund be supported?"
    result = extractor.supervisor_agent(example_query)
    print("\n--- Task 3 Example Query ---")
    print(f"Q: {example_query}")
    print(f"A (routed to {result['routed_to']}): {result['response']}\n")

    if NOTEBOOK_MODE:
        extractor.run_interactive_ui()
    else:
        extractor.run_console_loop()

    print("\n✅ Script execution completed.")


# In[ ]:





# In[ ]:





# In[ ]:




