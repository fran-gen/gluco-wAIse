import json
import os

# Data
kb_data = [
    {
        "questions": [
            "What are good snacks for people with diabetes?",
            "What should I eat between meals if I have diabetes?",
            "Can you recommend some healthy diabetic snacks?",
            "What are the best snack options for diabetics?",
            "Are there low-sugar snacks that are good for people with diabetes?"
        ],
        "answer": "Healthy snack options include Greek yogurt, almonds, boiled eggs, and vegetables with hummus."
    },
    {
        "questions": [
            "How often should a person with diabetes eat?",
            "Is it better to eat frequently with diabetes?",
            "Should diabetics eat every few hours?",
            "How many meals a day should I eat as a diabetic?",
            "What’s the eating schedule for someone with diabetes?"
        ],
        "answer": "It’s recommended to eat small meals every 3 to 4 hours to help maintain stable blood sugar levels."
    },
    {
        "questions": [
            "Can I eat fruit if I have diabetes?",
            "What fruits are safe for diabetics?",
            "Is fruit okay for a diabetic diet?",
            "Are bananas or apples good for diabetics?",
            "Which fruits should I choose with diabetes?"
        ],
        "answer": "Yes, fruits like berries, apples, and oranges are good options due to their fiber content. Just watch portion sizes."
    },
    {
        "questions": [
            "What foods should I avoid as a diabetic?",
            "Are there any foods I should stay away from with diabetes?",
            "Which foods raise blood sugar too much?",
            "What should diabetics not eat?",
            "What are some unhealthy foods for people with diabetes?"
        ],
        "answer": "Avoid sugary drinks, processed snacks, white bread, and high-sugar desserts."
    },
    {
        "questions": [
            "Is it okay to skip meals if I have diabetes?",
            "What happens if I miss a meal with diabetes?",
            "Can skipping meals cause problems for diabetics?",
            "Should I avoid skipping meals as a diabetic?",
            "Is fasting without eating dangerous for diabetes?"
        ],
        "answer": "Skipping meals can lead to low blood sugar. It's better to eat regularly and plan meals carefully."
    },
    {
        "questions": [
            "What is the best breakfast for someone with diabetes?",
            "Can you suggest a diabetic-friendly breakfast?",
            "What should I eat for breakfast as a diabetic?",
            "Are eggs and whole grains good for diabetic breakfast?",
            "What’s a healthy breakfast if I have diabetes?"
        ],
        "answer": "A balanced breakfast with protein, fiber, and healthy fats—like eggs, whole grain toast, and avocado—is a great choice."
    },
    {
        "questions": [
            "Can I eat carbs if I have diabetes?",
            "Are carbohydrates bad for diabetics?",
            "What kind of carbs can I eat with diabetes?",
            "Should I avoid all carbs with diabetes?",
            "How can diabetics safely eat carbs?"
        ],
        "answer": "Yes, but focus on complex carbs like whole grains, legumes, and vegetables, and control portions."
    },
    {
        "questions": [
            "How can I control my blood sugar through diet?",
            "What foods help manage blood sugar?",
            "What’s the best diet to keep blood sugar stable?",
            "How can I lower my blood sugar with food?",
            "Any nutrition tips to maintain blood sugar levels?"
        ],
        "answer": "Eat balanced meals, avoid refined carbs and sugars, monitor carb intake, and stay hydrated."
    },
    {
        "questions": [
            "Is intermittent fasting safe for diabetics?",
            "Can people with diabetes try intermittent fasting?",
            "Is fasting good or bad for blood sugar control?",
            "Should diabetics do time-restricted eating?",
            "Is it risky to fast with diabetes?"
        ],
        "answer": "It depends on the individual and medication. Always consult a healthcare provider before starting any fasting regimen."
    },
    {
        "questions": [
            "What drinks are safe for diabetics?",
            "Can I drink juice with diabetes?",
            "What beverages are good for blood sugar control?",
            "Are sugar-free drinks safe for diabetics?",
            "What should I drink instead of soda if I have diabetes?"
        ],
        "answer": "Water, unsweetened tea, and black coffee are generally safe. Avoid sugary drinks and limit fruit juice."
    }
]

# Save location
os.makedirs("drafting-agent/data/kb", exist_ok=True)
file_path = "drafting-agent/data/kb/diabetes_kb.json"

# Save JSON
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(kb_data, f, indent=2)
