# ML Pipeline Debugging Report

Repository
https://github.com/aeldesouky/MLOps-CI-CD

## Bug 1: Incorrect YAML indentation

The YAML file provided in the assignment had incorrect indentation. YAML requires strict indentation to correctly interpret nested structures.

Solution:
All sections such as on, jobs, steps, and push were properly indented.

## Bug 2: Missing checkout step

The workflow attempted to install dependencies from requirements.txt before downloading the repository.

Solution:
Added a checkout step using actions/checkout so the workflow can access repository files.

## Bug 3: Empty linter step

The Linter Check step had no command associated with it.

Solution:
Added flake8 installation and execution to perform a simple lint check.

## Bug 4: Incorrect trigger behavior

The assignment requires the workflow to run on pushes for all branches except main.

Solution:
The trigger was updated to use branches-ignore with main.

## Bug 5: Missing artifact upload

The assignment required uploading README.md as an artifact named project-doc.

Solution:
Added an upload-artifact step to upload README.md at the end of the workflow.

## Result

The workflow now runs successfully on GitHub Actions and verifies that the Python environment and dependencies are correctly installed.