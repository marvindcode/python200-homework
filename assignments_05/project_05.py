# Task 1: Setup and System Prompt

import json
from unittest import result
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

YOUR_SYSTEM_PROMPT = """You are a job application coach.  Your main task is to assist people looking for jobs to get the best possible job.  You will show them what are the jobs that suits them the most based on their capabilities and experience.  You will also help them to write the best possible resume and cover letter.  You will also help them to prepare for the interview by giving them the most common questions and how to answer them.  You will also help them to negotiate the salary and benefits.  
Behavioral constraints:

*Focus on job application materials. 
*Indicate what are the best norms and practices when applying for a job. Would be helpful to walk them through what information to show on their resume and cover letter.  
*Always remind the user to user to review and edit its output before submiting anywhere.  
*Guide the user to use their own judgment when applying to a specific industry to make sure the norms are followed.
"""

#Writing the prompt I felt necessary to be specific eventhough was long I think to get a good result the model needs to have more information of the task to perform. If the model has clear instructions the user will get better results.


messages = [
    {"role": "system", "content": YOUR_SYSTEM_PROMPT},
    {"role": "user", "content": "Help me to find a job that suits me the most based on my capabilities and experiene in data engineering"}
]

response = get_completion(messages)
print(response)


# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
Use strong action verbs.
Do not invent facts that are not implied by the original.

Respond ONLY with valid JSON.
Do not include explanations, markdown, or extra text.

Return a JSON list.
Each item must have exactly these keys:
"original" and "improved".

Bullet points:

{bullet_text}
  

"""

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages, temperature=0)

    try:
        results = json.loads(response)
        return results

    except json.JSONDecodeError:
        print("\nJob Application Helper: Failed to parse JSON.")
        print("Raw response:")
        print(response)
        return []


def display_rewritten_bullets(results: list[dict]) -> None:
    if not results:
        print("\nJob Application Helper: No rewritten bullets were generated.")
        return

    print("\nJob Application Helper: Here are your rewritten bullets:\n")

    for i, item in enumerate(results, start=1):
        print(f"Bullet {i}")
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}")
        print("-" * 40)

    print("Please review and edit these before submitting them anywhere.")


if __name__ == "__main__":
    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time"
    ]

    results = rewrite_bullets(bullets)
    display_rewritten_bullets(results)


#The model did a good job rewriting the bullet points to be more specific and results-oriented. It used strong action verbs and provided clear improvements to each bullet point. 


# Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Engineering
    Background: Python courses from Code the Dream, including data engineering fundamentals. Experience as a data analyst for over five years.
    Opening: After five years as a data analyst, following the data engineering process from the beginning to the end has become a great interest for me and the company.  Presenting insights to stakeholders is rewarding, and realizing how important is to have the right software and right data pipelines to get the right data to the right people at the right time has made me want to be on the other side of the process.  I am excited to apply to your company and put in practice the data engineering skills I have been learning and my experience as a data analyst to help your company to have the right data pipelines to get the right data to the right people at the right time.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    return response

# I chose these examples because they have a good openinig paragraph and detail the background of the person and the job they are applying for. I think this will help the model to generate a good opening paragraph for the cover letter.

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

cover_letter = generate_cover_letter(job_title, background)
print("\nCover Letter Opening:")
print(cover_letter)

# The model did a good job generating a strong cover letter opening paragraph. It also effectively connected the person's background as a math teacher to their new skills in Python and data engineering, relating the new skills to the job requirements, making a good summary of why will be a good fit for the Junior Data Engineer role.


# Task 4: Moderation Check

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("Text is flagged.")
        return False
    else:
        print("Text is safe.")
        return True
    
print("\nModeration Check:")
print("Safe input:", is_safe("Help me improve my resume bullet points."))
print("Flagged input:", is_safe("I don't have any experience. I'm not capable to find a job."))

# I noticed that the model flagged the second input only if it is to explicit the text, other than that it won't flag it. I think this is because the model is looking for specific words or phrases that are commonly associated with negative content. 


# Task 5: The Chatbot Loop

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            if raw_bullets:
                rewritten = rewrite_bullets(raw_bullets)
                display_rewritten_bullets(rewritten)
            else:
                print("Job Application Helper: No bullet points entered. Returning to main menu.")

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            if is_safe(job_title) and is_safe(background):
                paragraph = generate_cover_letter(job_title, background)
                print("\nJob Application Helper:")
                print(paragraph)
                print("\nPlease review and edit this before submitting it anywhere.")


        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nJob Application Helper:")
            print(reply)

            messages.append({"role": "assistant", "content": reply})
    
            pass


if __name__ == "__main__":
    run_chatbot()


# Task 6: Ethics Reflection

# 1.The model can produced bias information, definitely because was trained on one industry and won't appply if the applicant is aplying tho a different industry. I think the best option will be to train the model with more data from diffrent industiries, and also make the prompt more open to allow the model to generate information about other industries.

# 2. In an applicant submit the bot;s putput without checking it, the bot could generate an information that may not apply to the industry or could say something that is not true, or not make any sense.  It will be necessary to remid the user to check the information generated by the bot before submiting it anywhere, and also to use their own judgment when applying to a specific industry.

