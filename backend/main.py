import os
import re
import queue
import openai
import shutil
import tiktoken
import tempfile
import threading
import subprocess
import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from langchain.embeddings import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

if os.environ.get("USE_CHROMA", "false") == "true":
    from create_vector_db_chroma import embedding_search, embed_into_db
else:
    from create_vector_db import embedding_search, embed_into_db

load_dotenv()
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://frontend",
    "http://frontend:3000",
    *os.environ["ALLOWED_ORIGINS"].split(","),
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


encoder = tiktoken.get_encoding("cl100k_base")


def get_local_repo_path():
    if "LOCAL_REPO_PATH" in os.environ:
        print("Using LOCAL_REPO_PATH from environment variable")
        LOCAL_REPO_PATH = os.environ["LOCAL_REPO_PATH"]
    else:
        print("Using LOCAL_REPO_PATH from cache file")
        if os.path.exists(".talk-to-repo-cache"):
            with open(".talk-to-repo-cache", "r") as f:
                LOCAL_REPO_PATH = f.read()
        else:
            print("Creating new LOCAL_REPO_PATH")
            LOCAL_REPO_PATH = tempfile.mkdtemp()
    with open(".talk-to-repo-cache", "w") as f:
        f.write(LOCAL_REPO_PATH)
        os.environ["LOCAL_REPO_PATH"] = LOCAL_REPO_PATH
    print(f"Using LOCAL_REPO_PATH: {LOCAL_REPO_PATH}")
    return LOCAL_REPO_PATH


LOCAL_REPO_PATH = get_local_repo_path()


class RepoInfo(BaseModel):
    repo: str
    username: str
    hostingPlatform: str
    token: Optional[str] = None


class Message(BaseModel):
    text: str
    sender: str


class ContextSystemMessage(BaseModel):
    system_message: str


class Chat(BaseModel):
    messages: list


class CodeDiffs(BaseModel):
    diff: str


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)


def create_tempfile_with_content(content):
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def format_context(docs, LOCAL_REPO_PATH):
    corpus_summary = pd.read_csv("data/corpus_summary.csv")
    aggregated_docs = {}
    for d in docs:
        document_id = d.metadata["document_id"]
        start_location = (
            str(d.metadata["start_line"]),
            str(d.metadata["start_position"]),
        )
        end_location = (str(d.metadata["end_line"]), str(d.metadata["end_position"]))
        if document_id in aggregated_docs:
            aggregated_docs[document_id]["content"].append(d.page_content)
            aggregated_docs[document_id]["segments"].append(
                (start_location, end_location)
            )
        else:
            aggregated_docs[document_id] = {
                "content": [d.page_content],
                "segments": [(start_location, end_location)],
            }
    context_parts = []
    for i, (document_id, data_parts) in enumerate(aggregated_docs.items()):
        content_parts = data_parts["content"]
        context_segments = data_parts["segments"]
        entire_file_token_count = corpus_summary.loc[
            corpus_summary["file_name"] == document_id
        ]["n_tokens"].values[0]
        content_parts_token_count = sum(
            [len(encoder.encode(cp)) for cp in content_parts]
        )
        if content_parts_token_count / entire_file_token_count > 0.5:
            with open(LOCAL_REPO_PATH + "/" + document_id, "r") as f:
                file_contents = f.read()
            context_parts.append(
                f"[{i}] Full file {document_id}:\n" + add_line_numbers(file_contents)
            )
        else:
            for i in range(len(context_segments)):
                context_parts.append(
                    f"[{i}] this segment contains text from line {context_segments[i][0][0]} in position {context_segments[i][0][1]}  \n to line {context_segments[i][1][0]} and position {context_segments[i][1][1]}"
                    + f" of file {document_id}:\n {add_line_numbers(content_parts[i], start = context_segments[i][0][0])}"
                    + "\n---\n"
                )
    return "\n\n".join(context_parts)


def repl(m):
    repl.cnt += 1
    return f"{repl.cnt:03d}: "


def add_line_numbers(text, start=0):
    repl.cnt = int(start) - min(1, int(start))
    return re.sub(r"(?m)^", repl, text)


def format_query(query, context):
    return f"""Relevant context: {context}

    {query}"""


def extract_key_words(query):
    prompt = f"Extract from the following query the key words, \
        which will be used to grep a codebase. \
        Return the key words as a comma-separated list. \
        Query: {query}"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return (
        openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5,
        )
        .choices[0]
        .text.strip()
    )


def get_last_commits_messages(repo_path, n=20):
    result = subprocess.run(
        [
            "git",
            "-C",
            repo_path,
            "log",
            f"-n {n}",
            "--pretty=format:%s%n%n%h %cI%n----%n",
            "--name-status",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise Exception(
            f"Error getting commit messages: {result.stderr.decode('utf-8')}"
        )
    return "\n".join(result.stdout.decode("utf-8").split("\n"))


def get_last_commit(repo_path):
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise Exception(
            f"Error getting the last commit hash: {result.stderr.decode('utf-8')}"
        )
    return result.stdout.decode("utf-8").strip()


@app.get("/health")
def health():
    return "OK"


@app.get("/")
def healthroot():
    return "OK"


@app.post("/system_message", response_model=ContextSystemMessage)
def system_message(query: Message):
    return dict(
        system_message="\n\n".join(
            [
                open("query-preamble.txt", "r").read().strip(),
                f"Context:\n{format_context(embedding_search(query.text, k = int(os.environ['CONTEXT_NUM'])), LOCAL_REPO_PATH)}",
                f"Grep Context:\n{grep_more_context(query)}",
                f"Commit messages:\n{get_last_commits_messages(LOCAL_REPO_PATH, 5)}",
            ]
        )
    )


def clear_local_repo_path():
    global LOCAL_REPO_PATH
    try:
        shutil.rmtree(LOCAL_REPO_PATH)
    except:
        pass
    LOCAL_REPO_PATH = tempfile.mkdtemp()
    with open(".talk-to-repo-cache", "w") as f:
        f.write(LOCAL_REPO_PATH)
        os.environ["LOCAL_REPO_PATH"] = LOCAL_REPO_PATH


def grep_more_context(query):
    key_words = extract_key_words(query)
    print(f"Key words: {key_words}")
    context_from_key_words = ""
    for keyword in key_words.split(","):
        keyword = keyword.strip()
        print(f"Working in directory: {LOCAL_REPO_PATH}")
        output = subprocess.run(
            ["git", "grep", "-C 5", "-h", "-e", keyword, "--", "."],
            cwd=LOCAL_REPO_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        context_from_key_words += output.decode("utf-8") + "\n\n"
        print(f"Context from key word: {context_from_key_words}")
    return context_from_key_words[:1000]


def get_llm(g):
    return ChatOpenAI(
        model_name=os.environ["MODEL_NAME"],
        verbose=True,
        streaming=True,
        callback_manager=AsyncCallbackManager([ChainStreamHandler(g)]),
        temperature=os.environ["TEMPERATURE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_organization=os.environ["OPENAI_ORG_ID"],
    )


def get_llm_anthropic(g):
    return ChatAnthropic(
        model="claude-v1-100k",
        verbose=True,
        streaming=True,
        max_tokens_to_sample=1000,
        callback_manager=AsyncCallbackManager([ChainStreamHandler(g)]),
        temperature=os.environ["TEMPERATURE"],
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    )


@app.post("/chat_stream")
async def chat_stream(chat: List[Message]):
    print("In")
    encoding_name = "cl100k_base"

    def llm_thread(g, prompt):
        try:
            if os.environ["USE_ANTHROPIC"] == "true":
                llm = get_llm_anthropic(g)
            else:
                llm = get_llm(g)
            encoding = tiktoken.get_encoding(encoding_name)
            if len(chat) > 2:
                system_message, latest_query = [chat[0].text, chat[-1].text]
                keep_messages = [system_message, latest_query]
                new_messages = []
                token_count = sum([len(encoding.encode(m)) for m in keep_messages])
                for message in chat[1:-1:2]:
                    token_count += len(encoding.encode(message.text))
                    if token_count > 750:
                        break
                    new_messages.append(message.text)
                query_messages = [system_message] + new_messages + [latest_query]
                query_text = "\n".join(query_messages)
                context_from_key_words = grep_more_context(latest_query)
                docs = embedding_search(query_text, k=5)
                context = format_context(docs, LOCAL_REPO_PATH)
                formatted_query = format_query(
                    latest_query, context + context_from_key_words
                )
            else:
                formatted_query = chat[-1].text
            system_message = SystemMessage(content=chat[0].text)
            latest_query = HumanMessage(content=formatted_query)
            messages = [latest_query]
            token_limit = int(os.environ["TOKEN_LIMIT"])
            num_tokens = len(encoding.encode(chat[0].text)) + len(
                encoding.encode(formatted_query)
            )
            for message in reversed(chat[1:-1]):
                num_tokens += 4
                num_tokens += len(encoding.encode(message.text))
                if num_tokens > token_limit:
                    break
                else:
                    new_message = (
                        HumanMessage(content=message.text)
                        if message.sender == "user"
                        else AIMessage(content=message.text)
                    )
                    messages = [new_message] + messages
            messages = [system_message] + messages
            llm(messages)
        finally:
            g.close()

    def chat_fn(prompt):
        g = ThreadedGenerator()
        threading.Thread(target=llm_thread, args=(g, prompt)).start()
        return g

    return StreamingResponse(chat_fn(chat), media_type="text/event-stream")


@app.post("/load_repo")
def load_repo(repo_info: RepoInfo):
    clear_local_repo_path()
    print(f"Loading repo: {repo_info.repo}")
    if repo_info.hostingPlatform == "github":
        if repo_info.token:
            REPO_URL = (
                f"https://{repo_info.token}@github.com"
                f"/{repo_info.username}/{repo_info.repo}.git"
            )
        else:
            REPO_URL = f"https://github.com/{repo_info.username}/{repo_info.repo}.git"
    elif repo_info.hostingPlatform == "gitlab":
        if repo_info.token:
            REPO_URL = (
                f"https://oauth2:{repo_info.token}@gitlab.com"
                f"/{repo_info.username}/{repo_info.repo}.git"
            )
        else:
            REPO_URL = f"https://gitlab.com/{repo_info.username}/{repo_info.repo}.git"
    elif repo_info.hostingPlatform == "bitbucket":
        if repo_info.token:
            REPO_URL = (
                f"https://x-token-auth:{repo_info.token}@bitbucket.org"
                f"/{repo_info.username}/{repo_info.repo}.git"
            )
        else:
            REPO_URL = (
                f"https://bitbucket.org/{repo_info.username}/{repo_info.repo}.git"
            )
    elif repo_info.hostingPlatform == "local":
        REPO_URL = repo_info.repo
    else:
        return {"status": "error", "message": "Invalid hosting platform"}
    os.environ["GITHUB_TOKEN"] = repo_info.token
    embed_into_db(REPO_URL, LOCAL_REPO_PATH)
    return {
        "status": "success",
        "message": "Repo loaded successfully",
        "last_commit": get_last_commit(LOCAL_REPO_PATH),
    }


def call_gpt3(query, max_tokens, n=1, temperature=0.0):
    return (
        openai.Completion.create(
            engine="text-davinci-003",
            prompt=query,
            max_tokens=max_tokens,
            n=n,
            stop=None,
            temperature=temperature,
        )
        .choices[0]
        .text.strip()
    )


def grep_file_from_snippet(snippet):
    snippet = snippet.split("\n")[0]
    query = (
        f"In this code snippet:\n\n{snippet}\n\nWhat's the full file name? "
        f"Please only provide the file path and nothing else."
    )
    return call_gpt3(query, 50)


def generate_commit_message():
    return call_gpt3(
        "Generate a commit message for changes in these(code snippets) files.", 20
    )


def apply_diff_to_file(diff: str, file_path: str) -> None:
    with open(LOCAL_REPO_PATH + "/temp.diff", "w") as file:
        file.write(diff)
        file.write("\n")
    print(f"Applying diff to file: {file_path}, CWD: {LOCAL_REPO_PATH}")
    print(diff)
    print("-------")
    result = subprocess.run(
        [
            "git",
            "apply",
            "temp.diff",
            "--unidiff-zero",
            "--inaccurate-eof",
            "--allow-empty",
            "--ignore-whitespace",
        ],
        cwd=LOCAL_REPO_PATH,
        capture_output=True,
    )
    if not result.returncode == 0:
        print(f"Error message: {result.stderr.decode('utf-8')}")
        raise Exception(
            f"Failed to apply diff to file: {file_path} , return code: {result.returncode}"
        )


def create_commit_from_diffs(diffs):
    for code_diff in diffs:
        diff = code_diff.diff
        file_path = grep_file_from_snippet(diff)
        apply_diff_to_file(diff, file_path)
    commit_message = generate_commit_message()
    subprocess.run(["git", "add", "."], cwd=LOCAL_REPO_PATH)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=LOCAL_REPO_PATH)
    return True


@app.post("/create_commit")
def create_commit(diffs):
    return {"status": "success" if create_commit_from_diffs(diffs) else "error"}
