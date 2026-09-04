# Agent instructions

## Python environment policy

- Never install Python packages globally, into the user site, into Conda
  `base`, or into an unrelated Conda environment.
- Use the dedicated Conda environment named `generate_amr` for every Python
  command, dependency installation, notebook kernel, and test run.
- Do not create or use a `venv` or `.venv` for this project.
- If the `generate_amr` environment does not exist, create it with Python 3.10:

  ```powershell
  conda create --name generate_amr python=3.10 pip
  ```

- Invoke commands through Conda explicitly so shell activation is not required:

  ```powershell
  conda run --name generate_amr python -m pip install -r requirements.txt
  conda run --name generate_amr python path\to\script.py
  ```

- Always install packages through `conda run --name generate_amr`. When pip is
  required, use `python -m pip`; do not call a global `pip` executable.
- If environment creation or package installation requires network access or
  elevated permission, request approval before proceeding. Do not fall back to
  a global installation.
- Do not commit Conda environment files or installed packages unless the user
  explicitly requests an environment specification such as `environment.yml`.
