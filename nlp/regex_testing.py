import re
import json
from pathlib import Path

def regex(text):
    output = {}
    empty, heading, target, tool = re.split(r"Heading is |, target is |, tool to deploy is ", text)
    output["heading"], output["target"], output["tool"] = "".join(number_map[word] for word in heading.split()), target, tool.strip('.')
    return output

# testing: import the data file, extract command, check json output with actual labels in the file, calculate accuracy


current_directory = Path(__file__).parent # creates a relative path to the current directory gpt5
file_path = current_directory / '..' / 'novice' / 'nlp.jsonl' # navigates to nlp.jsonl
file_path = file_path.resolve() # convert to absolute path

commands = []
extracted_commands = []

number_map = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9"
}

# extract the command
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        commands.append(data["transcript"])
        extracted_commands.append({"heading": data["heading"], "target": data["target"], "tool": data["tool"]})
        
accuracy_count = 0
length = len(commands)
           
for i in range(length):
    json_output = regex(commands[i])
    print(json_output)
    print(extracted_commands[i])
    if json_output == extracted_commands[i]:
        accuracy_count += 1
                                                          
accuracy = accuracy_count / length * 100
print(f"Accuracy: {accuracy:.02f}%")