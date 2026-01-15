# PDF Data Extractor for Morningstar Fund Performance

This tool extracts tabular data from PDF files containing Morningstar fund performance data and loads it into a PostgreSQL database.

## Prerequisites

- Python 3.x
- PostgreSQL database

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL database:**
   - Create a database.
   - Set the following environment variables to configure the database connection:
     ```
     DB_NAME="your_database_name"
     DB_USER="your_username"
     DB_PASSWORD="your_password"
     DB_HOST="your_host"
     DB_PORT="your_port"
     ```

## Usage

Run the `pdf_extractor.py` script with the path to the PDF file as an argument:

```bash
python pdf_extractor.py /path/to/your/file.pdf
```
