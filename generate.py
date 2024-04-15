import google.generativeai as genai
import time
import json
import pandas as pd
MODEL = 'gemini-pro'
import os
genai.configure(api_key=os.environ.get('API_KEY'))
model = genai.GenerativeModel(MODEL)

# open cases.json and read each case
# for each case, generate a conversation between a doctor and patient

with open('cases.json', 'r') as f:
    cases = json.load(f)

# initialize empty dataframe
df = pd.DataFrame(columns=['case', 'conversation'])

for i in range(len(cases)):

    PROMPT = "You will have a conversation between a doctor (must be named Assistant) and patient (must be named Human), focusing on differential diagnosis based on the following patient history." + cases[str(i+1)] + "Assistant: Ask 1 question about main symptom details, covering nature, location, severity, and onset (use details verbatim from symptoms section) Ask 1 question about relevant medical history (use antecedents section) State the top 2 potential diagnoses from the given differential diagnosis list, along with their associated probabilities Ask 1 follow-up question related to a symptom detail that could help differentiate between the two potential diagnoses  Recommend 1 next step for diagnosis (e.g. test, examination, etc.)  Patient: Respond with complete statements from the symptoms and antecedents sections verbatim If asked about any missing details, do not provide any new information The conversation should strictly utilize the information provided in the patient history sections. No new clinical details should be introduced. The dialog must end with the Assistant ' The dialogue must be in the following format: ###Assistant: What are your symptoms? ### Human: I have chest pain even at rest. The pain is a knife stroke in my upper chest, breast (R) and breast (L). On a scale of 0-10, the pain intensity is 7 and the pace it appeared is 9.### Assistant: Any relevant medical history?### Human: I have had a spontaneous pneumothorax. I smoke cigarettes. I have a chronic obstructive pulmonary disease. Some family members have had a pneumothorax.### Assistant: Based on your history, the two most likely diagnoses are unstable angina at 26.2% probability, and spontaneous pneumothorax at 6% probability. Does the pain increase or decrease with physical exertion?### Human: I have symptoms that increase with physical exertion but alleviate with rest.### Assistant: Let's get a chest X-ray to examine for a potential pneumothorax."

    MODEL = 'gemini-pro'
    # time.sleep(1)
    response = model.generate_content(PROMPT)
    conv = str(response.candidates[0].content.parts[0])[7:-1]
    # add case and conversation to dataframe, don't use append()
    df.loc[i] = [i+1, conv]

    print(str(response.candidates[0].content.parts[0])[7:-1])
    print('-----------------------------------')

# write dataframe to generated_data.csv
df.to_csv('generated_data.csv', index=False)
