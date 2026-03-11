import os
import atexit
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    CodeInterpreterTool,
    CodeInterpreterToolAuto,
)


# -----------------------------
# Global state
# -----------------------------
credential = None
project_client = None
openai_client = None
agent = None
conversation = None
uploaded_file = None


def initialize_agent():
    global credential, project_client, openai_client
    global agent, conversation, uploaded_file

    load_dotenv()

    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint:
        raise ValueError("PROJECT_ENDPOINT is not set in .env")
    if not model_deployment:
        raise ValueError("MODEL_DEPLOYMENT_NAME is not set in .env")

    script_dir = Path(__file__).parent
    file_path = script_dir / "data.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing data file: {file_path}")

    print(f"Using project endpoint: {project_endpoint}")
    print(f"Using model deployment: {model_deployment}")
    print(f"Loaded file: {file_path.name}")

    credential = DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
    )
    project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
    openai_client = project_client.get_openai_client()

    # Upload the data file and create Code Interpreter tool access
    with open(file_path, "rb") as f:
        uploaded_file = openai_client.files.create(file=f, purpose="assistants")

    print(f"Uploaded file: {uploaded_file.filename}")

    code_interpreter = CodeInterpreterTool(
        container=CodeInterpreterToolAuto(file_ids=[uploaded_file.id])
    )

    # Create agent
    agent = project_client.agents.create_version(
        agent_name="data-agent",
        definition=PromptAgentDefinition(
            model=model_deployment,
            instructions=(
                "You are an AI assistant for the University of Houston CASA system. "
                "Answer questions using the uploaded knowledge file. "
                "Use Python only when it helps analyze, summarize, or structure the data. "
                "Be concise, accurate, and helpful."
            ),
            tools=[code_interpreter],
        ),
    )
    print(f"Using agent: {agent.name} (version: {agent.version})")

    # Create a stateful conversation
    conversation = openai_client.conversations.create()
    print(f"Conversation created: {conversation.id}")


def ask_agent(user_prompt: str) -> str:
    global openai_client, conversation, agent

    if not user_prompt or not user_prompt.strip():
        return "Please enter a question."

    try:
        openai_client.conversations.items.create(
            conversation_id=conversation.id,
            items=[{"type": "message", "role": "user", "content": user_prompt}],
        )

        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
            input="",
        )

        if response.status == "failed":
            return f"Response failed: {response.error}"

        return response.output_text or "No response text returned."

    except Exception as e:
        return f"Error: {e}"


def chat_agent(message, history):
    answer = ask_agent(message)
    return answer


def cleanup():
    global openai_client, project_client, credential, agent, conversation

    try:
        if openai_client is not None and conversation is not None:
            openai_client.conversations.delete(conversation_id=conversation.id)
            print("Conversation deleted")
    except Exception as e:
        print(f"Cleanup warning (conversation): {e}")

    try:
        if project_client is not None and agent is not None:
            project_client.agents.delete_version(
                agent_name=agent.name,
                agent_version=agent.version,
            )
            print("Agent deleted")
    except Exception as e:
        print(f"Cleanup warning (agent): {e}")

    try:
        if project_client is not None:
            project_client.close()
    except Exception:
        pass

    try:
        if credential is not None:
            credential.close()
    except Exception:
        pass


def main():
    initialize_agent()
    atexit.register(cleanup)

    demo = gr.ChatInterface(
        fn=chat_agent,
        title="UH CASA Assistant",
        description=(
            "Ask questions about CASA testing centers, tutoring, ETS, scheduling, "
            "rules, hours, locations, and parking."
        ),
        textbox=gr.Textbox(
            placeholder="Example: How do I schedule a CASA exam?",
            label="Your question",
        ),
    )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
