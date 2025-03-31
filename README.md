# TDS_Project-2
This project is due on 31 Mar 2025 EoD IST. Results will be announced by 15 Apr 2025.

For questions, use this Discourse thread.

Background
You are a clever student who has joined IIT Madras’ Online Degree in Data Science. You have just enrolled in the Tools in Data Science course.

To make your life easier, you have decided to build an LLM-based application that can automatically answer any of the graded assignment questions.

Specifically, you are building and deploying an API that accepts any question from one of these 5 graded assignments:

Graded Assignment 1
Graded Assignment 2
Graded Assignment 3
Graded Assignment 4
Graded Assignment 5
… and responds with the answer to enter in the assignment.

Create an API
Your application exposes an API endpoint. Let’s assume that it is at https://your-app.vercel.app/api/.

The endpoint must accept a POST request, e.g. POST https://your-app.vercel.app/api/ with the question as well as optional file attachments as multipart/form-data.

For example, here’s how anyone can make a request:

curl -X POST "https://your-app.vercel.app/api/" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Download and unzip file abcd.zip which has a single extract.csv file inside. What is the value in the "answer" column of the CSV file?" \
  -F "file=@abcd.zip"
Copy to clipboardErrorCopied
The response must be a JSON object with a single text (string) field: answer that can be directly entered in the assignment. For example:

{
  "answer": "1234567890"
}
Copy to clipboardErrorCopied
Deploy your application
Deploy your application to a public URL that can be accessed by anyone. You may use any platform, including Vercel.

(If you use ngrok, ensure that it is running continuously until you get your results.)

Share your code
Create a new public GitHub repository
Add an MIT LICENSE file
Commit and push your code

Evaluation
Pre-requisites: Your repository MUST meet the following criteria to be eligible for evaluation
Your GitHub repository exists and is publicly accessible
Your GitHub repository has a LICENSE file with the MIT license
We will send 5 questions randomly chosen from the graded assignments above. Correct answers will be awarded 4 marks each.
Your score will be the sum of the marks above. No normalization. What you get is what you get.
