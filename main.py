import json
import pyperclip

from memory import *
from openai import OpenAI


# TODO:
# - Save thought history, so it doesn't go in circles doing the same thing
#     - Make thought history an LRU cache
# - Right now it only supports query -> response, no chat functionality
#     - Can possibly implement chat by replacing {query} with chat history
#       (so instead of just a user query, it is the full chat history of user
#       messages and AI messages)


MODEL = "gpt-4"
TOKEN_LIMIT = 1000
MESSAGE = """A user has asked the following query: "%s"

The file structure of the user's program is the following:
```
%s
```

You have already expanded some sections of the file structure. To solve the user's query, you may explore the file structure further, or if you think that you have everything you need, you can provide the solution in text form. You can expand parts of the file structure that are collapsed by entering in the number before the ellipses. Try to expand files as much as possible before solving the user's query, as it is best to explore the user's code in detail. Your answer should be formatted in one of two JSON formats. If you need to explore the file structure further, if the current file structure does not provide enough information to solve the query, or if you need more details about certain parts of the code to provide a better response, you can use the first JSON format. Here is an example of a message sent using the first format:
{"thoughts": your thoughts, "command": your command}

Your thoughts should be a string, containing a step-by-step approach to how you would solve the user's query. Explain what files and parts of files are useful to look at. After listing a few sections to explore, decide on one as the best option. Your command should be a single integer, corresponding to a section of the code that has been collapsed into ellipses. This field should only contain this number.

If you truly have everything necessary to solve the user's query, including all code snippets that are necessary to understand the user's code and explain to the user how they should solve their query, you can use the second JSON format:
{"message": your message}

Your message should explain in great detail to the user how they should solve their query. Provide code if that is relevant. Remember, you should only use the second format if you can precisely tell the user how to solve their query. If you are unsure where to edit in the code or what specifically to add, you should use the first command to explore the code more. Now, attempt to solve the query. To repeat, the user's query is "%s\""""


root_dir = "/Users/cravuri/Projects/MIT/21M.385/banger"
query = """I get this error: AttributeError: 'MidiInput' object has no attribute 'on_pdate' Can you fix this?"""

root = Directory(root_dir)
client = OpenAI()
root.open(root_dir)
while True:
    prev_size = root.size()
    while root.size() > TOKEN_LIMIT:
        root.close_oldest()
    cur_size = root.size()

    message = MESSAGE % (query, root.to_str(), query)
    print(root.to_str())
    pyperclip.copy(message)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": message}],
    )
    content = json.loads(response.choices[0].message.content)

    if "message" in content:
        print(content["message"])
        break
    elif "thoughts" in content and "command" in content:
        print(content["thoughts"])
        command = content["command"]
        root.open_by_index(command)
