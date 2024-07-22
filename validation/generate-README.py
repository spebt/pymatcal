import re
import os
import subprocess

jupyternb_list = []
for dirpath, dirnames, filenames in os.walk("./"):
    p = "^.+\.ipynb$"
    for filename in filenames:
        result = re.match(p, filename)
        if result != None:
            jupyternb_list.append(result.group(0))
    break

for jupyternb in jupyternb_list:
    subprocess.run(
        ["jupyter nbconvert --to markdown --output-dir=assets %s" % jupyternb],
        shell=True,
        check=True,
    )
    # break

markdown_list = []
for dirpath, dirnames, filenames in os.walk("assets"):
    p = "^.+\.md$"
    for filename in filenames:
        result = re.match(p, filename)
        if result != None:
            markdown_list.append(result.group(0))
    # break
output_list = []
for markdown_file in markdown_list:
    p = "!\[png\]\("
    with open("assets/" + markdown_file, "r") as file:
        output_list.append(re.sub(p, "![png](assets/", file.read()))

with open("README.md", "w") as file:
    file.write(
        "This file is generated with\n"
        + "```shell\n"
        + "python generate-README.md\n"
        + "```\n"
    )
for filename in markdown_list:
    if os.path.exists("assets/" + filename):
        os.remove("assets/" + filename)
with open("README.md", "a") as file:
    for output in output_list:
        file.write(output)
