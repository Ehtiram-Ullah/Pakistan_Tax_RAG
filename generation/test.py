from google import genai
import os
from dotenv import load_dotenv
import time
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
models = []

for model in client.models.list():
    models.append(model.name.replace("models/",""))


modelNum = 0
while(modelNum<len(models)):
        
    try:
        response = client.models.generate_content(
            model=models[modelNum],
            contents="Say hello in one sentence."
        ) 
        print( f"{models[modelNum]} Passed")
        print(response.text)
        break
    except Exception as error:

        last_error = error

        print(
            f"{models[modelNum]} failed "
            f"Error: {error}"
        )
        modelNum +=1
        time.sleep(2)


    

# print(response.text)