import json
from fuck_words import profanity


if __name__ == "__main__":
    text = input("""
  _________
(           )
|   O   O   |
|     <     |
|   \____/  |
 \         /
  =========     
                
Hey, its a NLP model wich was developed by BioCode team and it detectes profanity in chat! WoW!\n\n-So now you can write what ever you want in this space: """)
    censored_text = profanity.censor(text)
    output = {"input": text, "output": censored_text}
    with open("output.json", "w") as f:
        json.dump(output, f, indent=4)
    print("\nI have checked your inputs, so here is my evaluation: ", censored_text)
