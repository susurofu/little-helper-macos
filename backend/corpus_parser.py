from pathlib import Path
from queue import Queue
from datetime import datetime

import docx

class CorpusFolderIterator:
    "Iterates through a folder and its subfolders to find document files, ignoring temporary and system files."
    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)

    @staticmethod
    def _is_temporary_or_system_file(path: Path) -> bool:
        name = path.name.lower()

        # Common system files
        system_files = {
            ".ds_store",
            "thumbs.db",
            "desktop.ini",
        }

        if name in system_files:
            return True

        # Microsoft Office temporary / lock files
        if name.startswith("~$"):
            return True

        # Common temporary / backup files
        if name.startswith("~"):
            return True

        if name.endswith((".tmp", ".temp", ".bak", ".swp", ".swo")):
            return True

        # Hidden files
        if name.startswith("."):
            return True

        return False

    def get_document_files(self):
        document_extensions = {
            ".txt",
            ".pdf",
            ".doc",
            ".docx",
            ".odt",
            ".rtf",
            # ".xls",
            # ".xlsx",
            # ".xlsm",
            # ".ods",
            # ".csv",
            # ".ppt",
            # ".pptx",
            # ".odp",
        }

        files = [
            p
            for p in self.root_folder.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower() in document_extensions
                and not self._is_temporary_or_system_file(p)
            )
        ]
        return files


class CorpusParser:
    """
    parses the text of the document files from CorpusFolderIterator
    parse_all: parses all files and places them into a queque for further processing
    parse_file: parses a single file and returns the text
    also chunks docs depending on token size
    
    """

    def __init__(self, files, max_in_queue_size=100, enable_recording=False):
        self.files = files
        self.ready_to_process = Queue(maxsize=max_in_queue_size)
        self.max_in_queue_size = max_in_queue_size
        self.processed_files_n = 0
        self.total_files_n = len(files)
        self.enable_recording = enable_recording
        self.saved_texts = []
        self.logging = False


    def enable_logging(self, log_file:str):
        """
        requires txt file for logging
        """
        if ".txt" not in log_file:
            raise ValueError("The logging file should be in .txt format")
        
        self.logging =True
        self.log_file = log_file
        
        self.logged_files = None
        self.log_file_path = Path(self.log_file)
        

        if not self.log_file_path.exists():
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.touch(exist_ok=True)
            self.logs = []
        else:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                log_raw = f.read()
                splitted = log_raw.split(" ")
                if log_raw == "" or len(splitted) < 1:
                    raise TypeError("The format of log file is incorrect or corrupted. No logging is possible from this file.")
                self.logs = splitted



        # logg structure
        # doc path :!sep  chunk_number_ chunck_size, start_time, end_time
        pass


    def parse_all(self):


        if self.logging:
            processed_docs = [log_line.split(" :!sep ")[0] for log_line in self.logs]
            row_to_record = ""
            parsing_start_time = ""
            parsing_end_time = ""

            # requires enable logging before the method can run
            for file in self.files:
                            str_path = str(file)
                            if str_path in processed_docs:
                                continue

                            print(f"Parsing file: {file}")
                            parsing_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            text = self.parse_file(file)
                            if text:
                                print(text)
                                self.ready_to_process.put((file, text))
                            try:
                                self.processed_files_n += 1
                                parsing_end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                if self.enable_recording:
                                    print(f"Processed {self.processed_files.qsize()} out of {self.total_files} files.")
                                    self.saved_texts.append((text))
                            except Exception:
                                continue
        else:

            for file in self.files:
                print(f"Parsing file: {file}")
                text = self.parse_file(file)
                if text:
                    print(text)
                    self.ready_to_process.put((file, text))
                try:
                    self.processed_files_n += 1
                    if self.enable_recording:
                        print(f"Processed {self.processed_files.qsize()} out of {self.total_files} files.")
                        self.saved_texts.append((text))
                except Exception:
                    continue

    @staticmethod
    def parse_file(file):
        "parses a single file and returns the text"
        if file.suffix.lower() == ".txt":
            return parse_txt(file)
        elif file.suffix.lower() == ".pdf":
            return parse_pdf(file)
        elif file.suffix.lower() == ".docx":
            return parse_docx(file)
        elif file.suffix.lower() == ".doc":
            return parse_doc(file)
        elif file.suffix.lower() == ".odt":
            return parse_odt(file)
        elif file.suffix.lower() == ".rtf":
            return parse_rtf(file)
        else:
            print(f"Unsupported file type: {file.suffix}")
            return None



# parser functions for document types

def parse_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def parse_pdf(file_path):
    from pypdf import PdfReader #lazy import
    # Load the PDF file
    reader = PdfReader(file_path)
    # Extract and combine text from all pages
    full_text = ""
    try:
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            except Exception as e:
                print(f"Error extracting text from page in {file_path}: {e}")
                continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    return full_text

def parse_docx(file_path):
    "parses .docx or .doc files and returns the text"
    doc = docx.Document(file_path)
    
    # Extract and join text from all paragraphs
    try:
        full_text = [paragraph.text for paragraph in doc.paragraphs]
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    return '\n'.join(full_text)

def parse_doc(file_path):
    import textract

    # Extracts raw text from either .doc or .docx
    try:
        text = textract.process(file_path)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None

    res = text.decode("utf-8")
    return res

def parse_odt(file_path):
    import odf
    from odf import text, teletype

    textdoc = odf.opendocument.load(file_path)
    
    # Find all paragraph elements
    allparas = textdoc.getElementsByType(text.P)
    
    # Extract and join text from all paragraphs
    try:
        text_content = [teletype.extractText(p) for p in allparas]
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None
    return "\n".join(text_content)

def parse_rtf(file_path):
    from striprtf.striprtf import rtf_to_text

    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
        rtf_content = infile.read()

    # Convert the RTF markup to plain text
    try:
        plain_text = rtf_to_text(rtf_content)
    except Exception as e:
        print(f"Error converting RTF to text for {file_path}: {e}")
        return None
    
    return plain_text


