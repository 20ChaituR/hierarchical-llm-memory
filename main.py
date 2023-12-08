import argparse
import json
import os
import sys
import time

from memory import Directory
from openai import OpenAI

# TODO:
# - Use urwid for terminal output handling
# - Right now it only supports query -> response, no chat functionality
#     - Can possibly implement chat by replacing {query} with chat history
#       (so instead of just a user query, it is the full chat history of user
#       messages and AI messages)

MODEL = "gpt-4-1106-preview"
# MODEL = "gpt-3.5-turbo-1106"
FILE_TOKEN_LIMIT = 1000

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

Summarize what has happened so far. Explain if you are currently on the right track, or if anything unexpected happened. Be sure to question your assumptions, as the file and code structure may not be what you expect. Respond in plaintext, giving only the explanation and no JSON. To repeat, the user's query is "%s"."""
SUMMARY = """

You have already been exploring this code for some time, and the following is a summary of your previous thoughts and what you have done so far:
```
%s
```

Please give a single JSON response, in either the first format or the second format, based on the file structure and your previous thoughts."""

import curses

_num_lines = 0


def clear_and_print(stdscr, s: str) -> None:
    stdscr.clear()
    stdscr.scrollok(True)  # Enable scrolling in the window
    stdscr.idlok(True)  # Enable intelligent scrolling
    height, width = stdscr.getmaxyx()
    lines = s.split("\n")

    i = 0
    for line in lines:
        wrapped_lines = [
            line[i : i + width] for i in range(0, len(line), width)
        ]  # Wrap long lines
        for wrapped_line in wrapped_lines:
            if (
                stdscr.getyx()[0] >= height - 1
            ):  # Check if we're at the bottom of the window
                stdscr.scroll()  # Scroll up if at the bottom
                i -= 1
            stdscr.addstr(i, 0, wrapped_line)
            i += 1
    stdscr.refresh()


# def clear_and_print(stdscr, s: str) -> None:
#     stdscr.clear()
#     lines = s.split("\n")
#     for i, line in enumerate(lines):
#         stdscr.addstr(i, 0, line)
#     stdscr.refresh()

# global _num_lines
# for _ in range(_num_lines):
#     sys.stdout.write("\x1b[1A")
#     sys.stdout.write("\x1b[2K")

# count = len(s.split("\n"))
# _num_lines = count
# print(s)


def main(stdscr, root_dir, query):
    stdscr.clear()
    stdscr.refresh()

    root = Directory(root_dir)
    client = OpenAI()
    summary = f"To solve the user's query, I need to explore the file structure to find the relevant files. I currently only see the root directory, {os.path.basename(root_dir)}/, so I should expand that directory to explore it further."
    while True:
        message = MESSAGE % (query, root.to_str(), SUMMARY % summary)
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": message}],
        )
        assistant_message = response.choices[0].message.content
        # assistant_message = '{"command": 1}'
        content = json.loads(assistant_message)

        if "message" in content:
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()
            print(content["message"])
            # clear_and_print(stdscr, content["message"])
            break
        elif "command" in content:
            command = content["command"]
            root.open_by_index(command)
            while root.size() > FILE_TOKEN_LIMIT:
                root.close_oldest()

        user_response = RESPONSE % (root.to_str(), query)
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "text"},
            messages=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_message},
                {"role": "user", "content": user_response},
            ],
        )
        summary = response.choices[0].message.content
        # summary = query
        time.sleep(5)
        clear_and_print(stdscr, f"{root.to_str()}\nSUMMARY: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer any query about a codebase.")
    parser.add_argument(
        "root_dir", type=str, help="The root directory of your codebase."
    )
    parser.add_argument(
        "query", type=str, help="The question you have about your codebase."
    )

    args = parser.parse_args()

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    main(stdscr, args.root_dir, args.query)
    # main(args.root_dir, args.query)
