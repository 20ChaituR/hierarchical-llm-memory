import json
import pyperclip

from memory import *
from openai import OpenAI
from typing import List


# TODO:
# - Save thought history, so it doesn't go in circles doing the same thing
#     - Make thought history an LRU cache
# - Right now it only supports query -> response, no chat functionality
#     - Can possibly implement chat by replacing {query} with chat history
#       (so instead of just a user query, it is the full chat history of user
#       messages and AI messages)

MODEL = "gpt-4-1106-preview"
FILE_TOKEN_LIMIT = 1000
THOUGHT_TOKEN_LIMIT = 1000

MESSAGE = """A user has asked the following query: "%s"

The file structure of the user's program is the following:
```
%s
```

You have already expanded some sections of the file structure. To solve the user's query, you may explore the file structure further, or if you think that you have everything you need, you can provide the solution in text form. You can expand parts of the file structure that are collapsed by entering in the number before the ellipses. Try to expand files as much as possible before solving the user's query, as it is best to explore the user's code in detail. Your answer should be formatted in one of two JSON formats. If you need to explore the file structure further, if the current file structure does not provide enough information to solve the query, or if you need more details about certain parts of the code to provide a better response, you can use the first JSON format. Here is an example of a message sent using the first format:
{"command": your command}

Your command should be a single integer, corresponding to a section of the code that has been collapsed into ellipses. This field should only contain this number.

If you truly have everything necessary to solve the user's query, including all code snippets that are necessary to understand the user's code and explain to the user how they should solve their query, you can use the second JSON format:
{"message": your message}

Your message should explain in great detail to the user how they should solve their query. Provide code if that is relevant. Remember, you should only use the second format if you can precisely tell the user how to solve their query. If you are unsure where to edit in the code or what specifically to add, you should use the first command to explore the code more.%s"""
RESPONSE = """The following is the file structure after executing your given command:
```
%s
```

Please summarize what has happened so far. Explain if you are currently on the right track, or if anything unexpected happened. Be sure to question your assumptions, as the file and code structure may not be what you expect. To repeat, the user's query is "%s\""""
SUMMARY = """

You have already been exploring this code for some time, and the following is a summary of your previous thoughts and what you have done so far:
```
%s
```

Please give a single JSON response, in either the first format or the second format, based on the file structure and your previous thoughts."""


def num_tokens(messages: List) -> int:
    encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    count = 0
    for message in messages:
        tokens = encoding.encode(message)
        count += len(tokens)
    return count


root_dir = "/Users/cravuri/Projects/MIT/21M.385/banger"
query = """Right now, the IntroScreen button is slightly off-center. Can you tell me exactly what to do to make it exactly centered on screen?"""

root = Directory(root_dir)
client = OpenAI()
thoughts = [
    "To solve the user's query, I need to explore the provided file structure and find all the relevant files. So, I should open the root directory to see how the user's files are structured."
]
summary = "To solve the user's query, I need to explore the file structure to find the relevant files. So far, I have opened the root directory and observed the top level structure. Nothing unexpected has happened, and I should continue exploring the user's file structure."
root.open(root_dir)
while True:
    while num_tokens(thoughts) > THOUGHT_TOKEN_LIMIT:
        thoughts.pop(0)

    if summary is None:
        message = MESSAGE % (query, root.to_str(), "")
    else:
        message = MESSAGE % (query, root.to_str(), SUMMARY % summary)

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": message}],
    )
    content = json.loads(response.choices[0].message.content)

    print(message)
    print(content)

    if "message" in content:
        break
    elif "thoughts" in content and "command" in content:
        thoughts.append(content["thoughts"])
        command = content["command"]
        root.open_by_index(command)
        while root.size() > FILE_TOKEN_LIMIT:
            root.close_oldest()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.choices[0].message.content},
            {"role": "user", "content": RESPONSE % (root.to_str(), query)},
        ],
    )
    summary = response.choices[0].message.content
    print(RESPONSE % (root.to_str(), query))
    print(summary)
