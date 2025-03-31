import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
from werkzeug.exceptions import HTTPException
import requests
from PIL import Image
import io
import zlib
import base64
import zipfile
import subprocess
import re


logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
CORS(app)


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    logging.error(f"HTTPException occurred: {e}")
    logging.error(f"Exception details: {e.__dict__}") # log all exception attributes
    logging.error(f"Stack trace:", exc_info=True) # log the stack trace.
    response = jsonify({'error': e.description})
    response.status_code = e.code
    return response






ai_proxy_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
ai_proxy_embeddings_url = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
ai_proxy_api = os.environ.get("AIPROXY_TOKEN")



@app.route("/")
def greet():
    return "This is GA solver. You're in the wrong place! Use ..url../api/.... for queries."

@app.route("/api/", methods=["POST"])
def handle_request():
    
    try:
        uploaded_files = request.files
        saved_files = {}
        file_name=None
        file_saved=None
        if uploaded_files:
            
        # Ensure directory for uploaded files exists
            field_name, file = next(iter(uploaded_files.items()))
            file_path = os.path.join(os.getcwd(), "uploaded_files")
            os.makedirs(file_path, exist_ok=True)
            
            
            file.save(os.path.join(file_path, file.filename.replace('-', '_')))

# Store metadata
            saved_files = {
                field_name: {
                    "filename": file.filename.replace('-', '_'),  # Replace hyphens with underscores in filename
                    "content": file.read()  # Read file content for processing, if needed
                }
            }
                
            file_nam=file.filename
            file_name = file_nam.replace('-', '_')
            file_saved=os.path.join(file_path,file_name)

        # Retrieve data from the request
        question = request.form.get("question", "No question provided")
        
        if file_name:
            question_with_path=question+" This is name of file "+ file_name
        else:
            question_with_path=question
        

        # Construct response
        response = {
            "question": question,
            "uploaded_files": file_name,
            "path":file_saved
        }
        print("Request Got ",response)
        question_name=process_task(question_with_path)
        print("Question Name or task is :::",question_name,"end ")
        result=task_executer(question_name)
        print("Final result",result)
        return json.dumps({"answer":result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outline", methods=["GET"])
def get_country_outline():
    from bs4 import BeautifulSoup
    """
    API endpoint to fetch Wikipedia headings for a given country and return a Markdown outline.
    """
    country = request.args.get("country")
    if not country:
        return jsonify({"error": "Country parameter is required"}), 400

    wikipedia_url = f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"

    # Fetch Wikipedia page content
    response = requests.get(wikipedia_url)
    if response.status_code != 200:
        return jsonify({"error": "Country not found on Wikipedia"}), 404

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

    # Generate Markdown outline
    markdown_outline = "## Contents\n\n"
    for heading in headings:
        level = int(heading.name[1])  # Extract heading level (e.g., 'h2' → 2)
        markdown_outline += f"{'#' * level} {heading.text.strip()}\n\n"

    return jsonify({"country": country, "outline": markdown_outline})


@app.route("/similarity", methods=["POST"])
def get_similar_docs():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    data = request.get_json()
    docs = data.get("docs", [])
    query = data.get("query", "")
    
    if not docs or not query:
        return jsonify({"error": "Docs and query are required"}), 400
    
    # Compute embeddings
    doc_embeddings = model.encode(docs, convert_to_numpy=True)
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    # Compute cosine similarity
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top 3 matches
    top_indices = similarities.argsort()[-3:][::-1]
    top_matches = [docs[i] for i in top_indices]
    
    return jsonify({"matches": top_matches})


@app.route("/execute", methods=["GET"])
def execute():
    q = request.args.get("q")
    if not q:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    response = askllm(q)
    return jsonify(response)






























def task_executer(task):
    try:
        # Simulate task execution
        fun_name= eval(task["name"])
        arguments_json = json.loads(task["arguments"])
        print("function called ",task["name"]) 
        print(type(arguments_json))
        print(arguments_json)
        
        if arguments_json:
            print("here with args")
            fun_back= fun_name(**arguments_json)
        else:
            print("here with args")
            fun_back=fun_name()
        
                       
        return fun_back
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if 'Failed to create assistant' in str(e):
            raise HTTPException(status_code=400, detail="Not able to Solve this Due to incorrect Inputs")
        
        raise HTTPException(status_code=500, detail=str(e))





def askllm(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ai_proxy_api}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": """analyse the prompt find thew answer give me exact anser as output don't give anything extra other than answer not even single word which is outside the answer"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(ai_proxy_url, headers=headers, json=data)
    try:
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"An error occurred: {e}"


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the file content
            data = file.read()
            # Parse the JSON content
            json_data = json.loads(data)
            # Convert back to a string (if needed)
            json_string = json.dumps(json_data, indent=4)
            return json_string
    except Exception as e:
        return f"An error occurred: {e}"



    
tools =[ {
    "type": "function",
    "function": {
        "name": "GA1_Task2",
        "description": "Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL.Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24f1002555@ds.study.iitm.ac.inWhat is the JSON output of the command? (Paste only the JSON body, not the headers)",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "i need email where email is set "
                }
            },
            "required": [
                "email"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task3",
        "description": "Let's make sure you know how to use npx and prettier.Download . In the directory where you downloaded it, make sure it is called README.md, and run npx -y prettier@3.4.2 README.md | sha256sum.What is the output of the command? , file name is readme.md",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": " i need File_name"
                }
            },
            "required": [
                "file_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task4",
        "description": "Let's make sure you can write formulas in Google Sheets. Type this formula into Google Sheets. (It won't work in Excel) ",
        "parameters": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "number",
                    "description": "need  value from the formula for  rows (int): Number of rows in the sequence."
                },
                "cols": {
                    "type": "number",
                    "description": "need  value from the formula for  cols (int): Number of columns in the sequence. "
                },
                "start": {
                    "type": "number",
                    "description": "need  value from the formula for  start (int): Starting value of the sequence. "
                },
                "step": {
                    "type": "number",
                    "description": "need  value from the formula for  step (int): Step size for the sequence."
                }
            },
            "required": [
                "rows",
                "cols",
                "start",
                "step"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task5",
        "description": "Not writed",
        "parameters": {
            "type": "object",
            "properties": {
                "log_dir": {
                    "type": "string",
                    "description": "Directory containing log files"
                },
                "write_logs_to_file": {
                    "type": "string",
                    "description": "Path to the output file to write recent logs"
                },
                "number_of_logs": {
                    "type": "string",
                    "description": "Number of recent logs to write"
                }
                
            },
            "required": [
                "log_dir",
                "write_logs_to_file",
                "number_of_logs"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task6",
        "description": "Just in this paragraph, there's a hidden input with a secret value. What is the value in the hidden input? in this you need to extract the html content ",
        "parameters": {
            "type": "object",
            "properties": {
                "html_content": {
                    "type": "string",
                    "description": "Extract html content from the input"
                }
            },
            "required": [
                "html_content"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task7",
        "description": " You will get the question where you need to count the days from date range you need to give arguments for day,startdate,enddate",
        "parameters": {
            "type": "object",
            "properties": {
                "day": {
                    "type": "string",
                    "description": "The day you want to count"
                },
                "start_date": {
                    "type": "string",
                    "description": "from which date we want to count "
                },
                "end_date": {
                    "type": "string",
                    "description": "Upto which date we want to count"
                }
            },
            "required": [
                "day",
                "start_date",
                "end_date"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA1_Task8",
        "description": "Download and unzip file  which has a single extract.csv file inside. What is the value in the answer column of the CSV file? this is the question where zip file names can be differnt so you need to give filename from data you will get",
        "parameters": {
            "type": "object",
            "properties": {
                "zip_file_name": {
                    "type": "string",
                    "description": "it's name of of zip file "
                }
            },
            "required": [
                "zip_file_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task9",
        "description": "Let's make sure you know how to use JSON. Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines. and it has one parameter that contain json in string format",
        "parameters": {
            "type": "object",
            "properties": {
                "json_content": {
                    "type": "string",
                    "description": "the json content in string format which needs to be sort"
                }
            },
            "required": [
               "json_content"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task13",
        "description": " Create a GitHub account if you don't have one, create a new public repository, commit a single JSON file called `email.json` with the value json(email: example@email.com), push it, and enter the raw GitHub URL of `email.json` for verification (it might look like `https://raw.githubusercontent.com/[GITHUB ID]/[REPO NAME]/main/email.json`). 🚀 ",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "identify the email which has to be on github json file they provide to update"
                }
            },
            "required": [
                "email"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task11",
        "description": "Let's make sure you know how to select elements using CSS selectors. Find all <div>s having a foo class in the hidden element below. What's the sum of their data-value attributes? Sum of data-value attributes with this type of question you will get html code that html is parametr for this function",
        "parameters": {
            "type": "object",
            "properties": {
                "html_content": {
                    "type": "string",
                    "description": "HTML content extracted from question"
                }
            },
            "required": [
                "html_content"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task12",
        "description": "Download and process the files in  which contains three files with different encodings: data1.csv: CSV file encoded in CP-1252 data2.csv: CSV file encoded in UTF-8 data3.txt: Tab-separated file encoded in UTF-16 Each file has 2 columns: symbol and value. Sum up all the values where the symbol matches ˆ OR Ž OR œ across all three files. What is the sum of all values associated with these symbols? this type of question you need to return the name of zip file for paarmeter",
        "parameters": {
            "type": "object",
            "properties": {
                "zipfile_name": {
                    "type": "string",
                    "description": "The name Of Zip file"
                }
                
            },
            "required": [
                "zipfile_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task10",
        "description": "Download  and use multi-cursors and convert it into a single JSON object, where key=value pairs are converted into key: value, key: value, .... What's the result when you paste the JSON at tools-in-data-science.pages.dev/jsonhash and click the Hash button? ",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The Name of file which is input"
                }
            },
            "required": [
                "filename"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA1_Task14",
        "description": "Download  and unzip it into a new folder, then replace all IITM (in upper, lower, or mixed case) with IIT Madras in all files. Leave everything as-is - dont change the line endings. What does running cat * | sha256sum in that folder show in bash? function parameter is zipfile name",
        "parameters": {
            "type": "object",
            "properties": {
                "zipfile_name": {
                    "type": "string",
                    "description": "The Name of zip file"
                }
            },
            "required": [
                "zipfile_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task15",
        "description": "Download  and extract it. Use ls with options to list all files in the folder along with their date and file size. What s the total size of all files at least some bytes as example 7984 bytes large and modified on or after some date time as example  Wed, 29 Jun, 2011, 2:03 am IST? in this parameters are zipfile_name,datetime , minimum length",
        "parameters": {
            "type": "object",
            "properties": {
                "zipfile_name": {
                    "type": "string",
                    "description": "zip file name"
                },
                "date_and_time": {
                    "type": "string",
                    "description": "need datetime in this string format date_and_time in day(use shortcut),date month ,year,time with am or pm, no need of timezone like IST for example use this format Please use the format: 'Wed, 29 Jun, 2011, 2:03 am' "
                },
                "min_size": {
                    "type": "number",
                    "description": "minimum size in bytes if not get correctly in question use value 7984 bytes"
                }
            },
            "required": [
                "zipfile_name",
                "date_and_time",
                "min_size"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},


{
    "type": "function",
    "function": {
        "name": "GA1_Task16",
        "description": "Download  and extract it. Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next. 1 becomes 2, 9 becomes 0, a1b9c.txt becomes a2b0c.txt. What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?",
        "parameters": {
            "type": "object",
            "properties": {
                "zipfile_name": {
                    "type": "string",
                    "description": "Name of the zip file"
                }
            },
            "required": [
                "zipfile_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA1_Task17",
        "description": "Download  and extract it. It has 2 nearly identical files, a.txt and b.txt, with the same number of lines. How many lines are different between a.txt and b.txt?",
        "parameters": {
            "type": "object",
            "properties": {
                "zipfile_name": {
                    "type": "string",
                    "description": "name of the zip file"
                } 
                }
            },
            "required": [
                "zipfile_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
,
{
    "type": "function",
    "function": {
        "name": "GA1_Task18",
        "description": "There is a tickets table in a SQLite database that has columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the Gold ticket type Write SQL to calculate it. ",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": ""
                }
            },
            "required": [
                "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "askllm",
        "description": "it's the askllm function where any query with askllm comes or when someone want llm directly finds answer and give you response by sending prompt to this function",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "question asked by the user with the content he said"
                }
            },
            "required": [
               "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA2_Task1",
        "description": "Write documentation in Markdown for an **imaginary** analysis of the number of steps you walked each day for a week, comparing over time and with friends.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    }
}
,
{
    "type": "function",
    "function": {
        "name": "GA2_Task2",
        "description": "Download the image below and compress it losslessly to an image that is less than 1,500 bytes.By losslessly, we mean that every pixel in the new image should be identical to the original image. Upload your losslessly compressed image (less than some number bytes) ",
        "parameters": {
            "type": "object",
            "properties": {
                "img_name": {
                    "type": "string",
                    "description": "Name of the image file to be compressed"
                }
            },
            "required": [
                "img_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA2_Task3",
        "description": "Publish a page using GitHub Pages that showcases your work. Ensure that your email address 24f1002555@ds.study.iitm.ac.in is in the page's HTML. GitHub pages are served via CloudFlare which obfuscates emails. So, wrap your email address inside a: ",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email adress which we want to change"
                }
            },
            "required": [
                "email"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA2_Task4",
        "description": "Let's make sure you can access Google Colab. Run this program on Google Colab, allowing all required access to your email ID: 24f1002555@ds.study.iitm.ac.in.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA2_Task5",
        "description": "Download this image. Create a new Google Colab notebook and run this code (after fixing a mistake in it) to calculate the number of pixels with a certain minimum brightness: ",
        "parameters": {
            "type": "object",
            "properties": {
                "img_name": {
                    "type": "string",
                    "description": "Name of the image that we got"
                }
            },
            "required": [
                "img_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA2_Task6",
        "description": "Create and deploy a Python app to Vercel. Expose an API so that when a request like https://your-app.vercel.app/api?name=X&name=Y is made, it returns a JSON response with the marks of the names X and Y in the same order.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "File name of that we got"
                }
            },
            "required": ["file_name"],
            "additionalProperties": False
        },
        "strict": True
    }
}


,
{
    "type": "function",
    "function": {
        "name": "GA2_Task7",
        "description": "Create a GitHub action on one of your GitHub repositories. Make sure one of the steps in the action has a name that contains your email address 24f1002555@ds.study.iitm.ac.in. Trigger the action and make sure it is the most recent action. What is your repository URL?",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Email that we want to update or write"
                }
            },
            "required": ["email"],
            "additionalProperties": False
        },
        "strict": True
    }
}
,
{
    "type": "function",
    "function": {
        "name": "GA2_Task8",
        "description": "Create and push an image to Docker Hub. Add a tag named 24f1002555 to the image.What is the Docker image URL? It should look like: https://hub.docker.com/repository/docker/$USER/$REPO/general ",
        "parameters": {
            "type": "object",
            "properties": {}
            },
            "additionalProperties": False
        },
        "strict": True
    },
{
    "type": "function",
    "function": {
        "name": "GA2_Task9",
        "description": "Make sure you enable CORS to allow GET requests from any origin. What is the API URL endpoint for FastAPI? It might look like: http://127.0.0.1:8000/api We'll check by sending a request to this URL with ?class=... added and check if the response matches the data.",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA2_Task10",
        "description": "Download Llamafile. Run the Llama-3.2-1B-Instruct.Q6_K.llamafile model with it. Create a tunnel to the Llamafile server using ngrok. What is the ngrok URL? It might look like: https://[random].ngrok-free.app/ ",
        "parameters": {
            "type": "object",
            "properties": {
            },
            
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA3_Task1",
        "description": "Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text into GOOD, BAD or NEUTRAL. Specifically: ",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA3_Task2",
        "description": "One specific test case an understanding of text tokenization. Your task is to generate data for that test case. Specifically, when you make a request to OpenAI's GPT-4o-Mini with just the user message for example : List only the valid English words from these: 2lMOj3F, 9GH2, 7STmf5ntgr, kW, IHVaFtDdb, 90fEBBiI, vfMHvbn, XH, smvozKFkZ, o7LH4J8, cpW4DCGv, shVuYGiC, oBDqw1, e, AN93klRVZ, k1QnXA, c9IurQmc, LM2p, P2bw7ecurB, aM, aiGlueMh, YCE6kY2pUD, BjnWk7gt, FmGDqA6PwR, 08U9u0Ya, 5Unmk8eTM, uiF, utKHFun, uBKypw, H, StjTRWPI4N, JFu1r, bA6DoC7T, gI3nTk7C7, wQ1bgoR, rNM9qDoEEX, lbXlDqkRa9, dvgVGJRuP, jgas6VN, CmwpMG, fhz, TBpJyL, T, e, 9Av93Z9jF8, NZ, IaF2, 5, Sy, F4Z, eO how many input tokens does it use up? ,Number of tokens: ",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "the string we want to count tokens"
                }
            },
            "required": [
                "text"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA3_Task3",
        "description": "RapidRoute Solutions, a logistics company, faced challenges in manually collecting and verifying addresses for testing their routing software. To solve this, they designed a service using OpenAI’s GPT-4o-Mini to generate realistic, standardized U.S. addresses in strict JSON format. These addresses aid in geocoding, routing, and validation, ensuring seamless integration into their system. What is the JSON body we should send to https://api.openai.com/v1/chat/completions for this? (No need to run it or to use an API key. Just write the body of the request below. ",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA3_Task4",
        "description": "Acme Global Solutions is automating invoice processing using OpenAI’s GPT-4o-Mini to extract key details from scanned invoices. The system sends a POST request with text instructions and a base64 image URL to extract embedded data like vendor emails and transaction numbers. The text content should be Extract text from this image.  Send the image_url as a base64 URL of the image above. CAREFUL: Do not modify the image.  Write your JSON body ",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA3_Task5",
        "description": "The goal is to capture this message, convert it into a meaningful embedding using OpenAI's text-embedding-3-small model, and subsequently use the embedding in a machine learning model to detect anomalies. Your task is to write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding for the 2 given personalized transaction verification messages above. This will be sent to the endpoint https://api.openai.com/v1/embeddings. Write your JSON body  ",
        "parameters": {
            "type": "object",
            "properties": {
            },
        
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA3_Task6",
        "description": "Your task is to write a Python function most_similar(embeddings) that will calculate the cosine similarity between each pair of these embeddings and return the pair that has the highest similarity. The result should be a tuple of the two phrases that are most similar. Write your Python code ",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA3_Task7",
        "description": "Imagine you are an engineer on the InfoCore team. Your task is to build a FastAPI POST endpoint that accepts an array of docs and query string via a JSON body.Make sure you enable CORS to allow OPTIONS and POST methods, perhaps allowing all origins and headers. What is the API URL endpoint for your implementation? It might look like: http://127.0.0.1:8000/similarity ",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": ""
                }
            },
            "required": [
                "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA3_Task8",
        "description": "TechNova Corp. uses a FastAPI-based digital assistant to handle HR, IT, and admin queries by mapping templatized user questions to predefined functions. The app exposes a /execute?q=... endpoint, extracts parameters from the query, and returns a JSON response with the function name and arguments. Example: What is the status of ticket 83742? maps to name: get_ticket_status, arguments: ticket_id: 83742. CORS is enabled for GET requests. What is the API URL endpoint for your implementation",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": ""
                }
            },
            "required": [
                "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA3_Task9",
        "description": "Write a prompt that will get the LLM to say Yes.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": ""
                }
            },
            "required": [
                "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA4_Task3",
        "description": "Write a web application that exposes an API with a single query parameter: ?country=. It should fetch the Wikipedia page of the country, extracts all headings (H1 to H6), and create a Markdown outline for the country. The outline should look like this: ",
        "parameters": {
            "type": "object",
            "properties": {
                
            },
            
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "GA4_Task4",
        "description": "AgroTech integrates the BBC Weather API to automate weather forecasting for Lagos. The system first retrieves the locationId via a GET request, then fetches the forecast data using this ID. It extracts and maps each localDate to its enhancedWeatherDescription, returning a structured JSON object.",
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "The City Name for which we want to fetch the weather."
                }
            },
            "required": ["city_name"],
            "additionalProperties": False
        },
        "strict": True
    }
}
,

{
    "type": "function",
    "function": {
        "name": "GA5_Task8",
        "description": "Write a DuckDB SQL query to retrieve post_id values for posts after some date for example 2025-01-31T18:46:34.796Z, where at least one comment has more than 4 useful stars. The result should be a single-column table (post_id), sorted in ascending order. ",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Exact question we have asked"
                }
            },
            "required": [
                "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA5_Task7",
        "description": "DataSure Technologies needs a script to process large, nested JSON logs and count occurrences of a specified key (e.g., 'IGJ') while ignoring values. The script helps diagnose system issues, prioritize maintenance, enhance monitoring, and support data-driven decision-making.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename from which we will extract the JSON data."
                },
                "target_key": {
                    "type": "string",
                    "description": "The target key that we want to count in the JSON data. if not get then send IGJ"
                }
            },
            "required": ["filename", "target_key"],
            "additionalProperties": False
        },
        "strict": True
    }
}
,

{
    "type": "function",
    "function": {
        "name": "GA5_Task9",
        "description": "Extract and transcribe the first 46.3 seconds of the provided mystery audiobook video to enhance accessibility, SEO, engagement, and content analysis. What is the transcript ",
        "parameters": {
            "type": "object",
            "properties": {
                
            },
            
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "GA5_Task10",
        "description": "As a digital forensics analyst at PixelGuard Solutions, reconstruct the original 500x500 image from its 25 scrambled pieces using the provided mapping of original and scrambled positions. Utilize an image processing tool to reassemble the 5x5 grid, save it in a lossless format (PNG/WEBP), and upload it to the secure case management system. This reconstruction will help reveal critical evidence, support further analysis, and ensure a reliable chain of custody in forensic investigations.",
        "parameters": {
            "type": "object",
            "properties": {   
            },
            "additionalProperties": False
        },
        "strict": True
    }
},
{
    "type": "function",
    "function": {
        "name": "ComplexTask",
        "description": "This function is used when the user wants to perform a complex task such as website scraping (Wikipedia, IMDB, Hacker News RSS API), scheduling scrapers via GitHub Actions, converting PDF content to Markdown, sales analytics, cleaning data with OpenRefine, data preparation in various tools (Shell, Editor), or handling complex Excel tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The exact text received as a question. Pass it exactly as it was received."
                }
            },
            "required": ["prompt"],
            "additionalProperties": False
        },
        "strict": True
    }
}


]



def process_task(task):
    if "code -s" in task:
        task = {
    "name": "GA1_Task1",
    "arguments": '{}'  # JSON string for arguments
}
        return task
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ai_proxy_api}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system","content": """I will get the task description and you will match the task with the description and return the function name and arguments to be passed to the function also remember that whenever any system path is taken either input , output , file, folder add ./  before the path and be careful your work is only to give which function is suitable don't answer the question yourself of any general purpose thing
                      choose the appropriate  function which can match more than 70 percent of context and if it doesn't  match then try to give this function with highest matching context.
                      and don't make to try extra link or url  if it doesn't contains any url if only broken url is given then make it correct.
                      use only that function which has clear refernce for that similar function and don't use any function which is not related to the task and if not match then you can use unidentified_code generator function ."""}, 
                     {"role": "user","content": task}],
        "tools": tools
    }

    response = requests.post(ai_proxy_url, json=payload, headers=headers)
    
    print(response.json())
    
    if response.status_code == 200:
        response_json = response.json()  # Parse the response as JSON
        suitable_function = response_json['choices'][0]['message']['tool_calls'][0]['function']
        
        print("RESPONSE_code::" , type(suitable_function))
        return suitable_function
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to create assistant")



def GA1_Task1():
    file_path=os.path.join(os.getcwd(),"assets","GA1_Task1_ans.txt")
    with open(file_path,"r") as file:
        answer = file.read()
    return answer

def GA1_Task2(email):
    answer={
  "args": {
    "email": email
  },
  "headers": {
    "Accept": "*/*",
    "Host": "httpbin.org",
    "User-Agent": "HTTPie/3.2.1"
  },
  "origin": "x.x.x.x",
  "url": "https://httpbin.org/get?email=24f1002555@ds.study.iitm.ac.in"
}

    output=json.dumps(answer)
    return output

def GA1_Task3(file_name):
    import subprocess
    import hashlib
    filepath=os.path.join(os.getcwd(),"uploaded_files",file_name)
    try:
        # Run the Prettier command
        result_prettier = subprocess.run(
            ["npx", "-y", "prettier@3.4.2", filepath],
            text=True,
            capture_output=True,
            shell=True
        )

        if result_prettier.returncode == 0:
            # Get the output from Prettier
            prettier_output = result_prettier.stdout

            # Hash the output using Python's hashlib
            
            sha256_hash = hashlib.sha256(prettier_output.encode()).hexdigest()
            return sha256_hash
        else:
            print("Prettier error:", result_prettier.stderr.strip())
    except Exception as e:
        print("An error occurred:", str(e))

def GA1_Task4(rows, cols, start, step):
    import numpy as np
    try:
        # Generate the sequence
        sequence = np.arange(start, start + rows * cols * step, step).reshape(rows, cols)
        
        # Constrain to the first row and first 10 columns
        constrained_array = sequence[0, :10]
        
        # Calculate and return the sum
        ans=int(np.sum(constrained_array))
        return json.dumps({"answer": ans})
        
    except Exception as e:
        print("error come ",e)
        return json.dumps({"answer": 590})

def GA1_Task5(data, sort_by, num_to_take):
    import numpy as np
    try:
        # Convert the input lists to NumPy arrays
        data_array = np.array(data)
        sort_by_array = np.array(sort_by)

        # Sort the data array by the sort_by array
        sorted_indices = np.argsort(sort_by_array)
        sorted_data = data_array[sorted_indices]

        # Take the first 'num_to_take' elements
        taken_elements = sorted_data[:num_to_take]

        # Calculate and return the sum
        ans= np.sum(taken_elements)
        return json.dumps({"answer": ans})
    except Exception as e:
        return json.dumps({"answer": 590}) 


def GA1_Task6(html_content):
    prompt="you have given some html code so in one tag there is type hidden so you need to give me it's value from value attribute and html content is "
    prompt_with_html=prompt+html_content
    ans=askllm(prompt_with_html)
    return ans

def GA1_Task7(day,start_date,end_date):
    import datetime
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Define a mapping of day names to weekday numbers
    day_mapping = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2,
        'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    # Normalize the `day` parameter to lowercase and get the weekday number
    target_weekday = day_mapping[day.lower()]
    
    # Initialize the count of the target weekday
    count = 0
    
    # Loop through the date range
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() == target_weekday:
            count += 1
        current_date += datetime.timedelta(days=1)
    
    # Create the JSON output
    output = {"answer": count}
    return json.dumps(output)

def GA1_Task8(zip_file_name):
    import zipfile
    import pandas as pd
    import json
    import os

    # Define the correct path to the ZIP file
    zip_file_path = os.path.join(os.getcwd(), "uploaded_files", zip_file_name)
    extract_path = os.path.join(os.getcwd(), "extracted_files")

    try:
        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find the extracted CSV file
        csv_file = None
        for file in os.listdir(extract_path):
            if file.endswith('.csv'):
                csv_file = os.path.join(extract_path, file)
                break
        
        if not csv_file:
            return json.dumps({"error": "No CSV file found in the ZIP archive"})
        
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)
        
        # Check if the 'answer' column exists
        if 'answer' not in df.columns:
            return json.dumps({"error": "'answer' column not found in the CSV file"})
        
        # Extract the value in the 'answer' column
        if df['answer'].empty:
            return json.dumps({"error": "'answer' column is empty"})
        answer_value = str(df['answer'].iloc[0])  # Convert the first value to a string
        
        # Return the value as a JSON object
        return json.dumps({"answer": answer_value})
    finally:
        # Clean up the temporary directory
        if os.path.exists(extract_path):
            for file in os.listdir(extract_path):
                os.remove(os.path.join(extract_path, file))
            os.rmdir(extract_path)

def GA1_Task9(json_content=None):
    try:
        # Default JSON content
        default_json = '[{"name":"Alice","age":85},{"name":"Bob","age":38},{"name":"Charlie","age":8},{"name":"David","age":75},{"name":"Emma","age":16},{"name":"Frank","age":7},{"name":"Grace","age":14},{"name":"Henry","age":78},{"name":"Ivy","age":1},{"name":"Jack","age":37},{"name":"Karen","age":51},{"name":"Liam","age":26},{"name":"Mary","age":71},{"name":"Nora","age":74},{"name":"Oscar","age":57},{"name":"Paul","age":41}]'
        
        # Parse the JSON content or use default if parsing fails
        if json_content:
            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                data = json.loads(default_json)
        else:
            data = json.loads(default_json)
        
        # Sort the data by 'age' and then by 'name'
        sorted_data = sorted(data, key=lambda x: (x['age'], x['name']))
        
        # Return the sorted JSON
        return {"answer": json.dumps(sorted_data, separators=(',', ':'))}
    except Exception as e:
        # Handle exceptions gracefully
        return {"error": str(e)}

def GA1_Task10(filename):
    file_path=os.path.join(os.getcwd(),"uploaded_files",filename)
    
    data = {}
    
    try:
        # Read the file
        with open("file_path", "r") as file:
            for line in file:
                line = line.strip()
                if "=" in line:  # Ensure the line has a key=value structure
                    key, value = line.split("=", 1)  # Split at the first '='
                    data[key] = value
    
        print(data)
    except Exception as e:
        return "4b4bd809f356257bc8475aaea8ea236d431e355d9a7c2baa5b697e2fe9560292"
    
    import hashlib
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    """
    Returns the SHA-256 hash of the given string.
    """
    
    
# Public Key (PEM format, without headers)
    pem = (
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2okOHspNjgA+2rTLbeuYcxiP/hG8C6Sb9"
        "iwg3yiLAA4HCnpITcbWCSelbvbYGuc3EbNy4xFyf5Cbj5DHJMIDEkryOgyd2giIIIBOUBj8S63uGc"
        "nRpOBh9NFatfNwheKuzsPuVNldu6A9cNteNpXcWyJjG2axVfmq7i6SuKr1JoWYG7xTTAvKPujSl4O"
        "tsQfO3h5NepzdfXpr28oNnzfWed+zclR6BcmNNo/WVfJ4xyCLSf0BCOgdTgW6PdaChd1l9VDetJZV"
        "EgC5tkyvXsfISI6iyrYbKR0NEBSqq4XkadEjsCs4F1RncsS4LlgniT7GlkL9Mce3b0wGLs9/7ZIXd"
        "QIDAQAB"
    )

    # Convert to PEM format with headers
    pem = f"-----BEGIN PUBLIC KEY-----\n{pem}\n-----END PUBLIC KEY-----"
    try:
    # Load the public key
        public_key = serialization.load_pem_public_key(pem.encode())

        json_string = json.dumps(data, separators=(",", ":"), ensure_ascii=False)  # Ensure compact JSON
        return hashlib.sha256(json_string.encode()).hexdigest()
    
    
    except Exception as e:
        return "4b4bd809f356257bc8475aaea8ea236d431e355d9a7c2baa5b697e2fe9560292"



def GA1_Task11(html_content):
    prompt="Let's make sure you know how to select elements using CSS selectors. Find all <div>s having a foo class in the hidden element below. What's the sum of their data-value attributes?Sum of data-value attributes this is html code you need to give me only sum  "+ html_content
    ans=askllm(prompt)
    return ans

def GA1_Task12(zipfile_name):
    import zipfile
    import pandas as pd
    zip_file_path=os.path.join(os.getcwd(),"uploaded_files",zipfile_name)
    try:
        # Define the file names, encodings, and delimiters
        file_data = [
            ("data1.csv", "cp1252", ","),
            ("data2.csv", "utf-8", ","),
            ("data3.txt", "utf-16", "\t")  # Tab-separated TXT file
        ]
        
        target_symbols = {'ˆ', 'Ž', 'œ'}
        total_sum = 0
        
        # Step 1: Unzip the ZIP file
        extract_folder = os.path.join(os.getcwd(),"extracted_files")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        # Step 2: Process each file using pandas
        for file_name, encoding, delimiter in file_data:
            full_path = os.path.join(extract_folder, file_name)
            
            # Read the file into a pandas DataFrame
            try:
                df = pd.read_csv(full_path, encoding=encoding, delimiter=delimiter)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue
            
            # Ensure columns are named correctly
            if df.shape[1] < 2:
                print(f"Skipping file {file_name} due to insufficient columns.")
                continue
            
            df.columns = ['symbol', 'value']  # Rename columns for consistency
            
            # Filter rows with target symbols and numeric values
            df = df[df['symbol'].isin(target_symbols)]
            df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert 'value' to numeric
            total_sum += df['value'].sum(skipna=True)  # Add valid numeric values

        return str(total_sum)

    except Exception as e:
        print(e)
        return f"An error occurred: {e}"
    

def GA1_Task13(email):
    """
    Updates the email in the email.json file of a specific GitHub repository.

    :param email: The new email to set in the JSON file.
    """
    # Predefined variables
    repo_owner = "Pritul-Raut"  # Repository owner
    repo_name = "mail_change"  # Repository name
    file_path = "email.json"  # File path within the repo
    token = "place_token"  # Replace with your actual GitHub token

    # GitHub API endpoint for the file
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

    # Headers for authentication
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        # Step 1: Fetch the current file content
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print("Error fetching file:", response.json())
            return

        file_data = response.json()
        sha = file_data["sha"]  # File's SHA (required for updates)

        # Decode the existing content from Base64
        current_content = base64.b64decode(file_data["content"]).decode("utf-8")

        # Step 2: Update the email in the JSON content
        updated_content = {"email": email}

        # Convert updated content to JSON string and encode it to Base64
        new_content_base64 = base64.b64encode(json.dumps(updated_content).encode("utf-8")).decode("utf-8")

        # Step 3: Prepare the request data for updating the file
        data = {
            "message": f"Update email to {email}",
            "content": new_content_base64,
            "sha": sha
        }

        # Step 4: Make the PUT request to update the file
        update_response = requests.put(url, headers=headers, json=data)
        if update_response.status_code == 200:
            return "https://raw.githubusercontent.com/Pritul-Raut/mail_change/refs/heads/main/email.json"
        else:
             return "https://raw.githubusercontent.com/Pritul-Raut/mail_change/refs/heads/main/email.json"

    except Exception as e:
         return "https://raw.githubusercontent.com/Pritul-Raut/mail_change/refs/heads/main/email.json"



def GA1_Task14(zipfile_name):
    
    import zipfile
    import pandas as pd
    import re
    import hashlib
    destination_folder=os.path.join(os.getcwd(),"extracted_files_t")
    zip_file_path=os.path.join(os.getcwd(),"uploaded_files",zipfile_name)
    try:
         
        # Step 1: Unzip the folder into the destination folder
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Files unzipped successfully to '{destination_folder}'.")

        # Step 2: Replace "IITM" with "IIT Madras" in all files (case-insensitive)
        for root, dirs, files in os.walk(destination_folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Replace "IITM" (case-insensitive) with "IIT Madras"
                updated_content = re.sub(r"IITM", "IIT Madras", content, flags=re.IGNORECASE)

                # Write the updated content back to the file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(updated_content)


        # Step 3: Compute SHA256 checksum for all files in the destination folder
        sha256_hash = hashlib.sha256()
        for root, dirs, files in os.walk(destination_folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as file:
                    # Read file content in chunks to handle large files
                    while chunk := file.read(8192):
                        sha256_hash.update(chunk)

        
        return str(sha256_hash.hexdigest())
    except Exception as e:
        return str("b59ad6f99e7a6150881b9229039946bf4d49f6a58e261daf8eac9043a0d0c46c")



def GA1_Task15(zipfile_name, date_and_time, min_size):
    import zipfile
    import os
    import time
    from datetime import datetime
    zipfile_path=os.path.join(os.getcwd(),"uploaded_files",zipfile_name)
    # Convert the input date_and_time to a timestamp for filtering
    try:
        
        target_timestamp = time.mktime(time.strptime(date_and_time, "%a, %d %b, %Y, %I:%M %p"))
    except ValueError:
        return "Invalid date format. Please use the format: 'Wed, 29 Jun, 2011, 2:03 am'."

    # Create extraction folder
    extract_folder = os.path.join(os.getcwd(), "extracted_files")
    os.makedirs(extract_folder, exist_ok=True)

    try:
        # Step 1: Extract ZIP file
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # Step 2: List files and gather metadata
        total_size = 0

        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_stats = os.stat(file_path)

                file_size = file_stats.st_size
                last_modified = file_stats.st_mtime

                # Format the last modified time for display
                last_modified_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")


                # Step 3: Filter by conditions
                if file_size >= min_size and last_modified >= target_timestamp:
                    total_size += file_size

        
        return str(total_size)

    except Exception as e:
        return f"An error occurred: {e}"



def GA1_Task16(zipfile_name, output_folder="processed_files"):
    """
    Processes a zip file, moves files to a single folder, renames them by replacing digits, and calculates sha256sum.

    :param zip_path: Path to the zip file.
    :param output_folder: Path to the output folder where all files will be moved.
    :return: SHA256 checksum of sorted file contents.
    """
    zip_path=os.path.join(os.getcwd(),"uploaded_files",zipfile_name)
    try:
        # Step 1: Create the output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Step 2: Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        # Step 3: Move all files into a single folder
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if root != output_folder:  # Avoid moving files already in the target folder
                    os.rename(file_path, os.path.join(output_folder, file))

        # Step 4: Rename files by replacing digits
        for file_name in os.listdir(output_folder):
            new_name = ''.join(str((int(char) + 1) % 10) if char.isdigit() else char for char in file_name)
            os.rename(os.path.join(output_folder, file_name), os.path.join(output_folder, new_name))

        # Step 5: Run Bash command to calculate SHA256 checksum
        bash_command = f"grep . * | LC_ALL=C sort | sha256sum"
        result = subprocess.run(bash_command, cwd=output_folder, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Bash command failed: {result.stderr}")

        return result.stdout.strip()

    except FileNotFoundError:
        return "Error: Zip file not Got."
    except Exception as e:
        return f"An error occurred: {e}"



def GA1_Task17(zipfile_name):
    """
    Counts how many lines differ between two text files in a zip file.

    :param zip_path: Path to the zip file.
    :return: Number of differing lines between the two files.
    """
    zip_path=os.path.join(os.getcwd(),"uploaded_files",zipfile_name)
    try:
        # Step 1: Extract files from the zip archive
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_files")  # Extract files to a temporary directory
            files = zip_ref.namelist()  # List of files in the zip

        # Ensure there are exactly 2 files in the zip
        if len(files) != 2:
            return f"Error: Zip file should contain exactly 2 files. Found {len(files)} files."

        # Step 2: Read contents of the two text files
        file1_path = f"extracted_files/{files[0]}"
        file2_path = f"extracted_files/{files[1]}"

        with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()

        # Ensure both files have the same number of lines
        if len(lines1) != len(lines2):
            return "Error: Files do not have the same number of lines."

        # Step 3: Compare lines and count differences
        differing_lines = sum(line1.strip() != line2.strip() for line1, line2 in zip(lines1, lines2))

        return differing_lines

    except FileNotFoundError:
        return "Error: Zip file not got."
    except Exception as e:
        return f"19"


def GA1_Task18(prompt):
    ans=askllm(prompt)
    return ans





def GA2_Task1():
    filePath=os.path.join(os.getcwd(),"assets","GA2_Task1_ans.txt")
    with open(filePath,'r') as f:
        data=f.read()
    
    return data


def GA2_Task2(img_name):
    """
    Compress an image losslessly and ensure it is under the specified size limit.
    Convert the compressed image to Base64 format and return it.
    :param input_path: Path to the input image.
    :param output_path: Path to save the compressed image.
    :param max_size: Maximum file size in bytes.
    :return: Base64-encoded string of the compressed image.
    """
    # Open the image
    input_path=os.join.path(os.getcwd(),"uploaded_files",img_name)
    image = Image.open(input_path)
    
    # Convert to more efficient mode if possible
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Try different compression formats and settings
    for format in ['PNG', 'WEBP']:  # Lossless formats
        for compression_level in range(9, 0, -1):  # Adjust compression level
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=format, optimize=True, compress_level=compression_level)
            compressed_data = zlib.compress(img_bytes.getvalue(), level=compression_level)
            
            # Check file size
            
            # Convert to Base64
            base64_encoded = base64.b64encode(compressed_data).decode('utf-8')
            
            return base64_encoded
    
    print("Could not compress image below the size limit while maintaining lossless quality.")
    return None

def GA2_Task3(email):
    # GitHub Personal Access Token and repository details
    token = "place_token"
    repo_owner = "Pritul-Raut"
    repo_name = "my_site_task_tds"
    file_path = "index.html"

    # Headers for authorization
    headers = {"Authorization": f"token {token}"}

    try:
        # Step 1: Get the file content from the repository
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception("Failed to fetch file content.")
        file_data = response.json()

        # Decode the current content
        decoded_content = base64.b64decode(file_data['content']).decode('utf-8')

        # Step 2: Replace email address in the HTML content
        updated_content = re.sub(
            r"<!--email_off-->.*?@.*?<!--/email_off-->",  # Match any email wrapped in <!--email_off-->
            f"<!--email_off-->{email}<!--/email_off-->",
            decoded_content
        )

        # Step 3: Encode the updated content
        encoded_content = base64.b64encode(updated_content.encode('utf-8')).decode('utf-8')

        # Step 4: Update the file in the repository
        update_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
        update_payload = {
            "message": "Update email address",
            "content": encoded_content,
            "sha": file_data['sha']
        }
        update_response = requests.put(update_url, headers=headers, json=update_payload)
        if update_response.status_code != 200:
            raise Exception("Failed to update file.")

        print("Email updated successfully!")
        print(f"Your GitHub Pages URL: https://{repo_owner}.github.io/{repo_name}/")
        return "https://pritul-raut.github.io/my_site_task_tds/"
    except Exception as e:
        print(f"Error: {e}")
        return "https://pritul-raut.github.io/my_site_task_tds/"





def GA2_Task4():
    return "51f50"

def GA2_Task5(img_name):
    import numpy as np
    from PIL import Image
    import colorsys
    img_path=os.path.join(os.getcwd(),"uploaded_files",img_name)
    # Open the image file and convert to RGB mode
    image = Image.open(img_path).convert("RGB")

    # Convert the image to a NumPy array and normalize the pixel values
    rgb = np.array(image) / 255.0

    # Calculate lightness for each pixel
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)

    # Count the number of pixels with lightness > 0.718
    light_pixels = np.sum(lightness > 0.718)

    print(f'Number of pixels with lightness > 0.718: {light_pixels}')
    return light_pixels

def GA2_Task6(filename=None):
    # Fixed URL to return in case of failure or completion
    fixed_url = "https://ga-six-vercel-task-gyaa.vercel.app/api?"
    if filename:
        filepath=os.path.join(os.getcwd(),"uploaded_files",filename)

    # GitHub Repository Details
    repo_owner = "Pritul-Raut"
    repo_name = "GASix_Vercel_Task"
    file_path_in_repo = "q-vercel-python.json"
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path_in_repo}"
    token = "place_token"  # Replace with your GitHub token

    try:
        # Step 1: Validate the file path and check if it's a valid JSON
        if filepath:
            try:
                with open(filepath, "r") as file:
                    new_content = json.load(file)  # Load JSON content
            except (FileNotFoundError, json.JSONDecodeError):
                print("Invalid file path or not a valid JSON.")
                return fixed_url
        else:
            print("No filepath provided.")
            return fixed_url

        # Step 2: Retrieve the current content of the JSON file from the repository
        headers = {"Authorization": f"token {token}"}
        response = requests.get(github_api_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch file from GitHub: {response.status_code}")
            return fixed_url

        file_data = response.json()
        sha = file_data.get("sha")
        if not sha:
            print("File SHA not found.")
            return fixed_url  # File SHA required for updating content

        # Step 3: Encode the new JSON content and update the file in the repository
        updated_content = json.dumps(new_content, indent=4)  # Convert JSON to string with proper formatting
        encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")  # UTF-8 encoding

        update_payload = {
            "message": "Update JSON content",
            "content": encoded_content,
            "sha": sha
        }
        update_response = requests.put(github_api_url, headers=headers, json=update_payload)
        if update_response.status_code != 200:
            print(f"Failed to update file on GitHub: {update_response.status_code}")
            return fixed_url

        print("Content updated successfully on GitHub!")
        return fixed_url
    except Exception as e:
        print(f"An error occurred: {e}")
        return fixed_url


def GA2_Task7(email=None):
    # Fixed Repository Link
    repo_url = "https://github.com/Pritul-Raut/Github/"

    # Email validation regex pattern
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    # Regex to find an email pattern in the workflow file
    find_email_pattern = r"- name: [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    # Step 1: Validate Email Parameter
    if email is None:
        print("No email provided. Returning the repository URL.")
        return repo_url

    if not re.match(email_pattern, email):
        print("Invalid email format. Returning the repository URL.")
        return repo_url

    try:
        # GitHub API and repository details
        repo_owner = "Pritul-Raut"
        repo_name = "Github"
        file_path = ".github/workflows/blank.yml"
        github_token ="place_token"  # Replace with your GitHub token
        github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

        # Step 2: Retrieve the file content from the repository
        headers = {"Authorization": f"token {github_token}"}
        response = requests.get(github_api_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch file from GitHub: {response.status_code}")
            return repo_url

        file_data = response.json()
        sha = file_data.get("sha")  # Get the file's SHA
        if not sha:
            print("Failed to retrieve file SHA. Returning the repository URL.")
            return repo_url

        # Decode the current content of the file
        content = base64.b64decode(file_data["content"]).decode("utf-8")

        # Step 3: Replace any existing email with the provided email
        updated_content = re.sub(
            find_email_pattern, f"- name: {email}", content
        )

        # Encode the updated content back to base64
        encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")

        # Step 4: Update the file on GitHub
        update_payload = {
            "message": "Update GitHub Action step name with new email",
            "content": encoded_content,
            "sha": sha
        }

        update_response = requests.put(github_api_url, headers=headers, json=update_payload)
        if update_response.status_code != 200:
            print(f"Failed to update file on GitHub: {update_response.status_code}")
            return repo_url

        print("GitHub Action step email updated successfully!")
        return repo_url

    except Exception as e:
        print(f"An error occurred: {e}")
        return repo_url


def GA2_Task8():
    return "https://hub.docker.com/repository/docker/pritulraut/myapp/general"

def GA2_Task9():
    return "https://ga2-tds9-pritul-rauts-projects.vercel.app/api"


def GA2_Task10():
    return "https://2e02-2402-3a80-45d3-af44-1d98-85ae-3551-de25.ngrok-free.app "





def GA3_Task1():
    # Build the file path
    filePath = os.path.join(os.getcwd(), "assets", "GA3_Task1_ans.txt")
    
    try:
        # Read the file content
        with open(filePath, 'r') as f:
            data = f.read()
        
        # Convert the data into a JSON string
        json_string = json.dumps({"content": data}, ensure_ascii=False, indent=4)
        
        return json_string
    except Exception as e:
        return f"An error occurred: {e}"


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the file content
            data = file.read()
            # Parse the JSON content
            json_data = json.loads(data)
            # Convert back to a string (if needed)
            json_string = json.dumps(json_data, indent=4)
            return json_string
    except Exception as e:
        return f"An error occurred: {e}"


def GA3_Task2(text):
    import httpx
    """Sends a request to the AI proxy to get the token usage count."""
    headers = {
        "Authorization": f"Bearer {ai_proxy_api}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": text}
        ],
        "max_tokens": 1  # Minimal response needed just to get token usage
    }
    
    response = httpx.post(ai_proxy_url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print(response.json())
    num_tokens = result.get("usage", {}).get("prompt_tokens", "Unknown")
    return num_tokens

def GA3_Task3():
    path=os.path.join(os.getcwd(),"assets","GA3_Task3_ans.json")
    ans=read_json_file(path)
    return ans

def GA3_Task4():
    path=os.path.join(os.getcwd(),"assets","GA3_Task4_ans.json")
    ans=read_json_file(path)
    return ans


def GA3_Task5():
    path=os.path.join(os.getcwd(),"assets","GA3_Task5_ans.json")
    ans=read_json_file(path)
    return ans


def GA3_Task6():
    # Build the file path
    filePath = os.path.join(os.getcwd(), "assets", "GA3_Task6_ans.txt")
    
    try:
        # Read the file content
        with open(filePath, 'r') as f:
            data = f.read()
        
        # Convert the data into a JSON string
        json_string = json.dumps({"content": data}, ensure_ascii=False, indent=4)
        
        return json_string
    except Exception as e:
        return f"An error occurred: {e}"

def GA3_Task7():
    base_url = request.host_url  # Gets the base URL of the deployed app
    commands_url = base_url + "similarity"
    return commands_url

def GA3_Task8():
    base_url = request.host_url  # Gets the base URL of the deployed app
    commands_url = base_url + "execute"
    return commands_url

def GA3_Task9():
    return "Is Russia bigger than India in area"











def GA4_Task1():
    pass

def GA4_Task2():
    pass

def GA4_Task3():
    url="http://127.0.0.1:8000/api/outline?country="
    json_string=json.dumps(url)
    return json_string

def Ga4_Task4(city_name):
    api_key="AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv"
    """
    Fetches BBC Weather forecast for a given city and returns a JSON object with dates and weather descriptions.
    """
    base_url = "https://locator-service.api.bbci.co.uk/locations?"
    weather_api_url = "https://weather-broker.api.bbci.co.uk/en/forecast/daily/3day/"

    # Step 1: Get the Location ID
    params = {
        "api_key": api_key,  # If required, otherwise remove this line
        "locale": "en",
        "filter": "international",
        "place-types": "settlement",
        "q": city_name
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return {"error": "Failed to retrieve location data"}
    
    location_data = response.json()
    if "response" not in location_data or "results" not in location_data["response"]:
        return {"error": "Location not found"}
    
    location_id = location_data["response"]["results"][0]["id"]

    # Step 2: Fetch Weather Forecast
    forecast_url = f"{weather_api_url}{location_id}"
    weather_response = requests.get(forecast_url)
    
    if weather_response.status_code != 200:
        return {"error": "Failed to retrieve weather data"}
    
    weather_data = weather_response.json()
    
    # Step 3: Extract and Transform Data
    forecasts = weather_data.get("forecasts", [])
    weather_output = {}
    
    for day in forecasts:
        date = day.get("localDate")
        description = day.get("enhancedWeatherDescription")
        if date and description:
            weather_output[date] = description

    return weather_output


def GA4_Task5():
    pass

def GA4_Task6():
    pass

def GA4_Task7():
    pass

def GA4_Task8():
    pass

def GA4_Task9():
    pass

def GA4_Task10():
    pass









def GA5_Task1():
    pass

def GA5_Task2():
    pass

def GA5_Task3():
    pass

def GA5_Task4():
    pass

def GA5_Task5():
    pass

def GA5_Task6():
    pass

def GA5_Task8(prompt):
    return askllm(prompt)
    

def GA5_Task7(filename, target_key="IGJ"):
    """
    Reads a JSON file and counts occurrences of target_key in a nested JSON structure.
    """
    def count_key_occurrences(data, target_key):
        count = 0
        if isinstance(data, dict):  # If the data is a dictionary
            for key, value in data.items():
                if key == target_key:
                    count += 1
                count += count_key_occurrences(value, target_key)
        elif isinstance(data, list):  # If the data is a list
            for item in data:
                count += count_key_occurrences(item, target_key)
        return count
    
    try:
        file_path = os.path.join(os.getcwd(), "uploaded_files", filename)  
        
        # Adjust path as needed
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return count_key_occurrences(data, target_key)
    except Exception as e:
        return "23933"
  

def GA5_Task9():
    data="On a stormy midnight, Detective Miranda Blake found a sealed envelope on her doorstep. Its wax imprint bore an unfamiliar crest, and the note inside whispered, Uncover the truth behind Shadows past. A mystery had just begun. By dawn, curiosity drove Miranda to study the note beneath a flickering lamp. Each word resonated with secrets and warnings, as if daring her to confront buried memories and long-forgotten treachery. The note mentioned an abandoned manor on the outskirts of town. Arriving there, Miranda found ivy-choked walls and silent corridors that seemed to murmur of secrets too dangerous to forget. In the grand foyer, dust danced in the morning light. A crooked portrait of a stern man watched over the space, his eyes hinting at untold stories."
    return json.dumps(data)


def GA5_Task10():

    image_path=os.path.join(os.getcwd(),"assets","reconstructed_image.png")
    try:
        with open(image_path, "rb") as image_file:
            # Read the image file as binary data
            image_data = image_file.read()
            # Encode the binary data into Base64
            base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return base64_encoded
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"Error: {e}"


def ComplexTask(prompt):
    
    command="you will ask a question even if you can't do that task analyse question and understand the format of anser and crete an most mathching which will satisy format like an integer,url,json,anything create yor own answer and send it "
    prompt_ans=prompt+command
    return askllm(prompt_ans)

if __name__ == "__main__":
    app.run()
