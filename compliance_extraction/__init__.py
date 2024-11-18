# Compliance Data Extraction //we should probably teach LM model to recognize them.
def extract_compliance(text):
    compliance_data = {}
    compliance_data["compliance_statements"] = extract_compliance_statements(text)
    compliance_data["CAS_numbers"] = extract_cas_numbers(text)
    compliance_data["regulated_substances"] = extract_regulated_substances(text)
    compliance_data["materials"] = extract_materials(text)
    Return compliance_data

# Utility Functions for Compliance Data Extraction
def extract_compliance_statements(text):
    compliance_statements = []
    compliance_keywords = ["compliance", "no regulated substances", "in accordance with"]
    For each sentence in text:
        If sentence contains any compliance keyword:
            Append sentence to compliance_statements
    Return compliance_statements

def extract_cas_numbers(text):
    Return Find all matches in text using CAS number regex pattern (format: XXXXX-XX-X)

def extract_regulated_substances(text):
    regulated_substances = []
    For each line in text:
        If line contains CAS number or matches regulated substance pattern:
            substance_name = extract_substance_name(line)
            Append {"name": substance_name, "CAS": CAS number} to regulated_substances list
    Return regulated_substances

def extract_substance_name(line):
    Return Extract the chemical name associated with CAS number

def extract_materials(text):
    material_keywords = ["Phenol", "Isopropylated Phosphate"]  # Add other materials as needed
    materials = []
    For each word in text:
        If word matches any material keyword:
            Append word to materials
    Return materials