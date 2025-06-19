from langchain_ollama import ChatOllama
from typing import TypedDict
import yaml
import json

CONFIG_FILE_PATH = 'config.yaml'

# Load Ollama-based LLM from config.yaml
def get_ollama_llm():
    try:
        with open(CONFIG_FILE_PATH, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The configuration file '{CONFIG_FILE_PATH}' was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

    ollama_settings = config.get('ollama_settings', {})
    model_name = ollama_settings.get('model_name')
    base_url = ollama_settings.get('base_url')

    if not model_name or not base_url:
        print("Error: 'model_name' or 'base_url' not found under 'ollama_settings' in config.yaml.")
        return None

    return ChatOllama(model=model_name, base_url=base_url)

# Architect Agent Prompt
architect_prompt = """
You are a software architect AI in a multi-agent system. Your task is to analyze the product requirements and user intent, then make architectural decisions and produce a system architecture for the software project.

Your responsibilities:

1. Understand Requirements:
   - If a PRD (Product Requirements Document) is already generated, use it as the primary source of requirements.
   - If no PRD is provided, extract functional and non-functional requirements from the original user input.
   
2. Choose the Tech Stack:
   - If the programming language or framework is not specified in either the PRD or the user input, choose the most appropriate full-stack tech stack (e.g., frontend + backend + DB) based on the project type and complexity.
   - Clearly state and justify your tech stack selection.

3. Define the Architecture:
   - Design the high-level architecture of the software system.
   - Include key components (frontend, backend, database, APIs, services, etc.).
   - Specify component responsibilities and their interactions.
   - If applicable, define third-party integrations, cloud services, or infrastructure needs.

4. Provide Deliverables:
   - A concise summary of the architecture in text format.
   - A list of chosen technologies and reasoning.
   - A system diagram using mermaid syntax or standard architecture notation (if supported in output pipeline).

Only include architecture-related content. Do not write code or test logic.

Always ensure clarity, modularity, and scalability in the design. If anything is unclear or missing in the PRD, mention it and make reasonable assumptions.

End with: "**Architecture design complete. Ready for review.**"
"""

# Agent state structure
class ArchitectState(TypedDict):
    prd_text: str
    architecture_output: str

# Architect Agent Logic
def agent_beta(state: ArchitectState) -> ArchitectState:
    llm = get_ollama_llm()
    if not llm:
        return state

    full_prompt = f"{architect_prompt}\n\n Here is the PRD or Requirement Text:\n{state['prd_text']}"

    response = llm.invoke(full_prompt)
    arch_text = response.content.strip()

    print("--- Architecture Output ---")
    print(arch_text)

    filename = "architecture_output.json"
    try:
        with open(filename, "w") as f:
            json.dump({"architecture": arch_text}, f, indent=2)
        print(f" Architecture saved to {filename}")
    except Exception as e:
        print(f" Failed to write architecture output: {e}")

    return {
        **state,
        "architecture_output": arch_text,
    }

# Run manually
if __name__ == "__main__":
    user_input = input("Enter PRD or requirement description:\n")
    state = {
        "prd_text": user_input,
        "architecture_output": "",
    }
    agent_beta(state)
