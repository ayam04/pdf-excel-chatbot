# PDF and Excel Chatbot

This project provides two separate tools to interact with PDF documents and CSV files using OpenAI's language model. One tool allows you to ask questions about the contents of a PDF, and the other enables interaction with a CSV file using natural language queries.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/ayam04/pdf-excel-chatbot.git
   cd pdf-excel-chatbot
   ```

2. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project root.
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_KEY=your_openai_api_key
     ```

## Usage

### PDF Chatbot

This tool extracts text from a PDF, embeds it into a vector store, and allows you to query the document using natural language.

1. **Update the PDF file path:**
   In `chat_pdf.py`, update the `pdf_path` variable with the path to your PDF file.
   ```python
   pdf_path = "path/to/your/pdf/file.pdf"
   ```

2. **Run the PDF chatbot:**
   ```sh
   python chat_pdf.py
   ```

3. **Interact with the chatbot:**
   - Ask any question related to the PDF content.
   - Type `exit` to quit.

### CSV Chatbot

This tool loads a CSV file into a SQLite database and allows you to interact with it using natural language queries.

1. **Update the CSV file path:**
   In `chat_csv.py`, update the `csv_path` variable with the path to your CSV file.
   ```python
   csv_path = "path/to/your/csv/file.csv"
   ```

2. **Run the CSV chatbot:**
   ```sh
   python chat_csv.py
   ```

3. **Interact with the chatbot:**
   - Ask any question related to the CSV data.
   - Type `exit` to quit.
