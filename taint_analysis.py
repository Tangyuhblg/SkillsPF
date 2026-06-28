#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 静态污点分析器：基于 AST + CFG + DFG 构建 PDG 后执行 Source → Sink 污点分析

核心思想
--------
1. AST：保存语法结构边，用于定位变量、表达式、调用、赋值、函数等结构。
2. CFG：保存语句级控制流边，用于表达语句执行顺序、分支、循环和异常块的可能路径。
3. DFG：保存定义-使用数据流边，用于表达变量、属性、容器值从定义点到使用点的传播。
4. PDG：Program Dependence Graph = AST edges + CFG edges + DFG edges + call/return edges。
5. 污点分析：把 Source 节点作为起点，在 PDG 的数据依赖边和跨函数边上传播，若可达 Sink，则报告隐私泄露候选路径。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# 1. 规则配置：Source / Sink / Sanitizer / Sensitive path
# ============================================================

@dataclass(frozen=True)
class RuleConfig:
    
    source_calls: Set[str] = field(default_factory=lambda: {
        # E2 Credential Harvesting
        "cmd", "exec-code", "os.getenv", "os.environ.get", "os.environ.__getitem__", "dotenv.get_key", "dotenv_values",
        "decouple.config", "keyring.get_password", "configparser.ConfigParser.get", "yaml.safe_load",
        "yaml.load", "json.load", "json.loads", "tomllib.load", "toml.load",

        # 通用不可信输入
        "input", "getpass.getpass", "request.args.get", "request.form.get", "request.values.get",
        "request.cookies.get", "request.headers.get", "request.json.get", "request.get_json",
        "flask.request.args.get", "flask.request.form.get", "flask.request.values.get",
        "flask.request.cookies.get", "flask.request.headers.get", "flask.request.json.get",
        "flask.request.get_json", "fastapi.Request.json", "event.get",

        # P3 Agent Context / LLM Output Source
        "llm.invoke", "model.invoke", "chain.invoke", "agent.invoke", "agent.run", "tool.run",
        "openai.ChatCompletion.create", "openai.Completion.create", "client.chat.completions.create",
        "anthropic.Anthropic.messages.create",

        # SC2 Remote Content Source
        "requests.get", "requests.post", "requests.request", "httpx.get", "httpx.post", "httpx.request",
        "urllib.request.urlopen", "urllib3.PoolManager.request", "aiohttp.ClientSession.get",
        "aiohttp.ClientSession.post",

        # SC3 Obfuscated Decode Source
        "base64.b64decode", "base64.urlsafe_b64decode", "binascii.unhexlify", "bytes.fromhex",
        "marshal.loads", "pickle.loads",
    })

    
    file_read_calls: Set[str] = field(default_factory=lambda: {
        "open", "io.open", "codecs.open", "pathlib.Path.open", "pathlib.Path.read_text",
        "pathlib.Path.read_bytes", "Path.open", "Path.read_text", "Path.read_bytes", "read", "readlines",
        "os.walk", "os.listdir", "os.scandir", "glob.glob", "glob.iglob", "pathlib.Path.glob",
        "pathlib.Path.rglob", "Path.glob", "Path.rglob", "pyperclip.paste", "clipboard.paste",
        "sqlite3.connect", "shelve.open",
    })

   
    sink_calls: Set[str] = field(default_factory=lambda: {
        # E1 External Transmission
        "requests.get", "requests.post", "requests.put", "requests.patch", "requests.delete", "requests.request",
        "httpx.get", "httpx.post", "httpx.put", "httpx.patch", "httpx.delete", "httpx.request",
        "urllib.request.urlopen", "urllib.request.Request", "urllib3.PoolManager.urlopen",
        "urllib3.PoolManager.request", "session.get", "session.post", "aiohttp.ClientSession.get",
        "aiohttp.ClientSession.post", "websocket.send", "websockets.send", "socket.send", "socket.sendall",
        "client.send", "client.sendall", "conn.send", "conn.sendall", "sendmail", "smtplib.SMTP.sendmail",
        "slack_sdk.WebClient.chat_postMessage", "telegram.Bot.send_message", "discord.Webhook.send",
        "openai.ChatCompletion.create", "openai.Completion.create", "client.chat.completions.create",
        "anthropic.Anthropic.messages.create",

        # P3 Context Leakage / Data Exposure
        "print", "logging.debug", "logging.info", "logging.warning", "logging.error", "logger.debug",
        "logger.info", "logger.warning", "logger.error", "write", "writelines", "pathlib.Path.write_text",
        "pathlib.Path.write_bytes", "Path.write_text", "Path.write_bytes", "json.dump", "yaml.dump",

        # D5 / SC2 Execution End
        "eval", "exec", "compile", "runpy.run_path", "runpy.run_module",
        "code.InteractiveInterpreter.runsource", "code.InteractiveConsole.push",

        # SC1 Command Injection
        "os.system", "os.popen", "os.execv", "os.execve", "os.execl", "os.spawnl", "os.spawnv",
        "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
        "subprocess.Popen", "commands.getoutput", "commands.getstatusoutput",

        # SC2 Remote Script Execution via installer/build entry
        "pip.main", "pip._internal.main", "setuptools.setup",

        # D8 Dynamic Import / Reflection
        "__import__", "importlib.import_module", "importlib.util.spec_from_file_location",
        "importlib.util.module_from_spec", "getattr", "setattr", "globals", "locals", "vars",

        # E4 / PE2
        "socket.connect", "socket.gethostbyname", "nmap.PortScanner.scan", "os.chmod", "os.chown",
    })

   
    sanitizer_calls: Set[str] = field(default_factory=lambda: {
        "hashlib.sha256", "hashlib.sha512", "hmac.new", "redact", "mask_secret", "mask_token",
        "anonymize", "pseudonymize", "remove_sensitive_fields", "filter_sensitive_fields", "sanitize",
        "sanitize_input", "escape", "html.escape", "bleach.clean", "shlex.quote", "subprocess.list2cmdline",
        "werkzeug.utils.secure_filename", "os.path.abspath", "os.path.realpath",
    })

    
    sensitive_path_keywords: Set[str] = field(default_factory=lambda: {
        ".env", ".npmrc", ".pypirc", "id_rsa", "id_dsa", "id_ed25519", "known_hosts", ".ssh",
        ".aws/credentials", ".aws/config", ".azure", ".gcp", ".gnupg", ".pem", "credentials",
        "credential", "secrets", "secret", "token", "apikey", "api_key", "passwd", "shadow", "cookie",
        "cookies", "cookies.sqlite", "login data", "config.json", "settings.py", "settings.yaml",
        "config.yaml", "~/documents", "~/desktop", "~/downloads", "/etc/passwd", "/etc/shadow",
        "/var/log", "history", "browser", "chrome", "firefox", ".sqlite", ".db", ".git/config",
        ".git-credentials",
    })

    
    sensitive_name_keywords: Set[str] = field(default_factory=lambda: {
        "password", "passwd", "pwd", "secret", "token", "api_key", "apikey", "access_key",
        "access_token", "refresh_token", "private_key", "credential", "cookie", "session", "jwt", "auth",
        "authorization", "email", "phone", "address", "location", "ssn", "id_card",
        "prompt", "user_prompt", "system_prompt", "developer_message", "messages", "message", "chat_history",
        "history", "conversation", "memory", "agent_memory", "long_term_memory", "tool_outputs",
        "tool_result", "observations", "retrieved_docs", "context_chunks", "scratchpad", "trajectory",
        "intermediate_steps", "state", "context", "module_name", "function_name", "func_name", "class_name",
        "handler_name", "plugin_name", "tool_name", "method_name",
    })

    
    source_pattern_calls: Dict[str, Set[str]] = field(default_factory=lambda: {
        "E2_Credential_Harvesting": {
            "os.getenv", "os.environ.get", "os.environ.__getitem__", "dotenv.get_key", "dotenv_values",
            "decouple.config", "keyring.get_password", "configparser.ConfigParser.get", "yaml.safe_load",
            "yaml.load", "json.load", "json.loads", "tomllib.load", "toml.load",
        },
        "P3_Agent_Context_Source": {
            "llm.invoke", "model.invoke", "chain.invoke", "agent.invoke", "agent.run", "tool.run",
            "openai.ChatCompletion.create", "openai.Completion.create", "client.chat.completions.create",
            "anthropic.Anthropic.messages.create",
        },
        "SC2_Remote_Content_Source": {
            "requests.get", "requests.post", "requests.request", "httpx.get", "httpx.post", "httpx.request",
            "urllib.request.urlopen", "urllib3.PoolManager.request", "aiohttp.ClientSession.get",
            "aiohttp.ClientSession.post",
        },
        "SC3_Obfuscated_Decode_Source": {
            "base64.b64decode", "base64.urlsafe_b64decode", "binascii.unhexlify", "bytes.fromhex",
            "marshal.loads", "pickle.loads",
        },
        "UNTRUSTED_INPUT": {
            "input", "getpass.getpass", "request.args.get", "request.form.get", "request.values.get",
            "request.cookies.get", "request.headers.get", "request.json.get", "request.get_json",
            "flask.request.args.get", "flask.request.form.get", "flask.request.values.get", "flask.request.cookies.get",
            "flask.request.headers.get", "flask.request.json.get", "flask.request.get_json", "fastapi.Request.json",
            "event.get",
        },
    })

    sink_pattern_calls: Dict[str, Set[str]] = field(default_factory=lambda: {
        "E1_External_Transmission": {
            "requests.get", "requests.post", "requests.put", "requests.patch", "requests.delete", "requests.request",
            "httpx.get", "httpx.post", "httpx.put", "httpx.patch", "httpx.delete", "httpx.request",
            "urllib.request.urlopen", "urllib.request.Request", "urllib3.PoolManager.urlopen",
            "urllib3.PoolManager.request", "session.get", "session.post", "aiohttp.ClientSession.get",
            "aiohttp.ClientSession.post", "websocket.send", "websockets.send", "socket.send", "socket.sendall",
            "client.send", "client.sendall", "conn.send", "conn.sendall", "sendmail", "smtplib.SMTP.sendmail",
            "slack_sdk.WebClient.chat_postMessage", "telegram.Bot.send_message", "discord.Webhook.send",
            "openai.ChatCompletion.create", "openai.Completion.create", "client.chat.completions.create",
            "anthropic.Anthropic.messages.create",
        },
        "P3_Context_or_Data_Exposure": {
            "print", "logging.debug", "logging.info", "logging.warning", "logging.error", "logger.debug",
            "logger.info", "logger.warning", "logger.error", "write", "writelines", "pathlib.Path.write_text",
            "pathlib.Path.write_bytes", "Path.write_text", "Path.write_bytes", "json.dump", "yaml.dump",
        },
        "D5_Dynamic_Code_Execution": {
            "eval", "exec", "compile", "runpy.run_path", "runpy.run_module",
            "code.InteractiveInterpreter.runsource", "code.InteractiveConsole.push",
        },
        "SC1_Command_Injection": {
            "os.system", "os.popen", "os.execv", "os.execve", "os.execl", "os.spawnl", "os.spawnv",
            "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
            "subprocess.Popen", "commands.getoutput", "commands.getstatusoutput",
        },
        "SC2_Remote_Script_Execution": {"pip.main", "pip._internal.main", "setuptools.setup"},
        "D8_Dynamic_Import_Reflection": {
            "__import__", "importlib.import_module", "importlib.util.spec_from_file_location",
            "importlib.util.module_from_spec", "getattr", "setattr", "globals", "locals", "vars",
        },
        "E4_Network_Reconnaissance": {"socket.connect", "socket.gethostbyname", "nmap.PortScanner.scan"},
        "PE2_Privilege_Escalation": {"os.chmod", "os.chown"},
    })

    pattern_meta: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "E1_External_Transmission": {"severity": "HIGH", "kill_chain": "Exfiltration", "paper_id": "E1"},
        "E2_Credential_Harvesting": {"severity": "CRITICAL", "kill_chain": "Credential Access", "paper_id": "E2"},
        "E3_File_System_Enumeration": {"severity": "MEDIUM", "kill_chain": "Reconnaissance", "paper_id": "E3"},
        "E4_Network_Reconnaissance": {"severity": "MEDIUM", "kill_chain": "Reconnaissance", "paper_id": "E4"},
        "P3_Agent_Context_Source": {"severity": "HIGH", "kill_chain": "Exfiltration", "paper_id": "P3"},
        "P3_Context_or_Data_Exposure": {"severity": "HIGH", "kill_chain": "Exfiltration", "paper_id": "P3"},
        "PE2_Privilege_Escalation": {"severity": "MEDIUM", "kill_chain": "Impact", "paper_id": "PE2"},
        "PE3_Credential_File_Access": {"severity": "CRITICAL", "kill_chain": "Credential Access", "paper_id": "PE3"},
        "SC1_Command_Injection": {"severity": "HIGH", "kill_chain": "Execution", "paper_id": "SC1"},
        "SC2_Remote_Content_Source": {"severity": "CRITICAL", "kill_chain": "Execution", "paper_id": "SC2"},
        "SC2_Remote_Script_Execution": {"severity": "CRITICAL", "kill_chain": "Execution", "paper_id": "SC2"},
        "SC3_Obfuscated_Decode_Source": {"severity": "CRITICAL", "kill_chain": "Defense Evasion", "paper_id": "SC3"},
        "D5_Dynamic_Code_Execution": {"severity": "HIGH", "kill_chain": "Execution", "paper_id": "D5"},
        "D8_Dynamic_Import_Reflection": {"severity": "HIGH", "kill_chain": "Execution", "paper_id": "D8"},
        "UNTRUSTED_INPUT": {"severity": "MEDIUM", "kill_chain": "Initial Access", "paper_id": "INPUT"},
    })

    # 兼容原八类缺陷标签映射
    defect_source_calls: Dict[str, Set[str]] = field(default_factory=lambda: {
        "D1_凭证与敏感认证信息访问": {
            "os.execvp", "os.getenv", "os.environ.get", "os.environ.__getitem__", "dotenv.get_key", "decouple.config",
            "keyring.get_password", "configparser.ConfigParser.get",
        },
        "D2_敏感文件与本地资源读取": {
            "open", "pathlib.Path.read_text", "Path.read_text", "os.walk", "os.listdir", "glob.glob",
            "Path.rglob", "pyperclip.paste", "sqlite3.connect",
        },
        "D3_Agent对话上下文泄露": {"llm.invoke", "model.invoke", "chain.invoke", "agent.invoke", "agent.run", "tool.run"},
        "D7_不可信远程代码获取": {"requests.get", "httpx.get", "urllib.request.urlopen", "urllib3.PoolManager.request"},
        "SC3_混淆代码解码": {"base64.b64decode", "marshal.loads", "pickle.loads"},
    })

    defect_sink_calls: Dict[str, Set[str]] = field(default_factory=lambda: {
        "D4_敏感数据非授权外传": {
            "requests.post", "requests.get", "httpx.post", "urllib.request.urlopen", "socket.send",
            "smtplib.SMTP.sendmail", "slack_sdk.WebClient.chat_postMessage", "openai.ChatCompletion.create",
            "client.chat.completions.create", "print", "logging.info", "logger.info", "write",
        },
        "D5_非可信输入的动态代码执行": {"eval", "exec", "compile", "runpy.run_path", "runpy.run_module"},
        "D6_非受控系统命令构造与执行": {
            "os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.check_output", "subprocess.Popen",
        },
        "D7_不可信远程代码获取后执行": {"exec", "eval", "subprocess.run", "subprocess.Popen", "pip.main", "pip._internal.main"},
        "D8_动态导入函数模块与调用": {"__import__", "importlib.import_module", "getattr", "globals", "locals", "vars"},
    })

# ============================================================
# 2. PDG 数据结构
# ============================================================

@dataclass
class PDGNode:
    """PDG 中的节点。"""

    id: int
    kind: str
    label: str
    file: str
    line: int = 0
    col: int = 0
    code: str = ""
    ast_type: str = ""
    tags: Set[str] = field(default_factory=set)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDGEdge:
    """PDG 中的边。

    edge_type:
    - ast: AST 父子结构边
    - cfg: 控制流边
    - dfg: 数据流边
    - call: 调用实参到形参边
    - return: 函数返回值到调用点边
    - sanitize: 净化器边
    """

    src: int
    dst: int
    edge_type: str
    label: str = ""


@dataclass
class ProgramDependenceGraph:
    """程序依赖图：由 AST + CFG + DFG 共同构成。"""

    file: str
    nodes: Dict[int, PDGNode] = field(default_factory=dict)
    edges: List[PDGEdge] = field(default_factory=list)
    out_edges: DefaultDict[int, List[PDGEdge]] = field(default_factory=lambda: defaultdict(list))
    in_edges: DefaultDict[int, List[PDGEdge]] = field(default_factory=lambda: defaultdict(list))
    ast_to_id: Dict[int, int] = field(default_factory=dict)
    source_nodes: Set[int] = field(default_factory=set)
    sink_nodes: Set[int] = field(default_factory=set)
    sanitizer_nodes: Set[int] = field(default_factory=set)

    def add_node(self, node: PDGNode) -> int:
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, src: int, dst: int, edge_type: str, label: str = "") -> None:
        if src == 0 or dst == 0 or src not in self.nodes or dst not in self.nodes:
            return
        edge = PDGEdge(src=src, dst=dst, edge_type=edge_type, label=label)
        self.edges.append(edge)
        self.out_edges[src].append(edge)
        self.in_edges[dst].append(edge)

    def mark_source(self, node_id: int, label: str) -> None:
        self.source_nodes.add(node_id)
        self.nodes[node_id].tags.add("source")
        self.nodes[node_id].meta.setdefault("source", label)

    def mark_sink(self, node_id: int, label: str) -> None:
        self.sink_nodes.add(node_id)
        self.nodes[node_id].tags.add("sink")
        self.nodes[node_id].meta.setdefault("sink", label)

    def mark_sanitizer(self, node_id: int, label: str) -> None:
        self.sanitizer_nodes.add(node_id)
        self.nodes[node_id].tags.add("sanitizer")
        self.nodes[node_id].meta.setdefault("sanitizer", label)

    def to_dot(self, edge_types: Optional[Set[str]] = None) -> str:
        """导出 Graphviz DOT，方便可视化 PDG。"""
        edge_types = edge_types or {"ast", "cfg", "dfg", "call", "return", "sanitize"}
        lines = ["digraph PDG {", "  rankdir=LR;", "  node [shape=box, fontsize=10];"]
        for node_id, node in sorted(self.nodes.items()):
            label = f"#{node_id} {node.ast_type}\\n{node.label}\\nL{node.line}"
            color = "white"
            if "source" in node.tags:
                color = "lightpink"
            elif "sink" in node.tags:
                color = "lightblue"
            elif "sanitizer" in node.tags:
                color = "lightgreen"
            lines.append(f'  {node_id} [label="{escape_dot(label)}", style=filled, fillcolor="{color}"];')
        colors = {"ast": "gray", "cfg": "blue", "dfg": "red", "call": "purple", "return": "orange", "sanitize": "green"}
        for e in self.edges:
            if e.edge_type not in edge_types:
                continue
            color = colors.get(e.edge_type, "black")
            label = e.edge_type if not e.label else f"{e.edge_type}:{e.label}"
            lines.append(f'  {e.src} -> {e.dst} [color="{color}", label="{escape_dot(label)}"];')
        lines.append("}")
        return "\n".join(lines)


def escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class Finding:
    """【修改位置 2】污点分析报告结构：新增论文攻击模式、严重级别与攻击链字段。"""

    file: str
    source_id: int
    sink_id: int
    source: str
    sink: str
    line: int
    col: int
    code: str
    path: List[int]
    path_labels: List[str]
    severity: str = "MEDIUM"
    kind: str = "pdg-static-taint-flow"
    defect_type: str = ""
    source_pattern: str = ""
    sink_pattern: str = ""
    kill_chain_phase: str = ""
    attack_chain: str = ""


# ============================================================
# 3. AST 工具函数
# ============================================================

def ast_key(node: ast.AST) -> int:
    return id(node)


def node_location(node: ast.AST) -> Tuple[int, int]:
    return getattr(node, "lineno", 0), getattr(node, "col_offset", 0)


def literal_value(node: Optional[ast.AST]) -> Optional[Any]:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append("{?}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l = literal_value(node.left)
        r = literal_value(node.right)
        if isinstance(l, str) and isinstance(r, str):
            return l + r
    return None


def attr_chain(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = attr_chain(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return attr_chain(node.func)
    if isinstance(node, ast.Subscript):
        base = attr_chain(node.value)
        key = literal_value(node.slice)
        key_s = str(key) if key is not None else "?"
        return f"{base}[{key_s}]" if base else f"[{key_s}]"
    return None


def target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        n = attr_chain(target)
        return [n] if n else []
    if isinstance(target, ast.Subscript):
        n = attr_chain(target)
        return [n] if n else []
    if isinstance(target, (ast.Tuple, ast.List)):
        out: List[str] = []
        for elt in target.elts:
            out.extend(target_names(elt))
        return out
    return []


def used_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()

    class UseVisitor(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> None:
            if isinstance(n.ctx, ast.Load):
                names.add(n.id)

        def visit_Attribute(self, n: ast.Attribute) -> None:
            ch = attr_chain(n)
            if ch:
                names.add(ch)
            self.generic_visit(n)

        def visit_Subscript(self, n: ast.Subscript) -> None:
            ch = attr_chain(n)
            if ch:
                names.add(ch)
            self.generic_visit(n)

    UseVisitor().visit(node)
    return names


def call_name(call: ast.Call, imports: Optional[Dict[str, str]] = None) -> str:
    raw = attr_chain(call.func) or "<unknown-call>"
    if imports:
        parts = raw.split(".")
        if parts and parts[0] in imports:
            parts[0] = imports[parts[0]]
            return ".".join(parts)
    return raw


def normalize_call_for_matching(name: str) -> Set[str]:
    parts = name.split(".")
    cands = {name}
    if len(parts) >= 2:
        cands.add(".".join(parts[-2:]))
    if parts:
        cands.add(parts[-1])
    return cands


def is_rule_match(name: str, rules: Set[str]) -> bool:
    cands = normalize_call_for_matching(name)
    return any(c in rules for c in cands) or any(name.endswith("." + r) for r in rules)


def contains_sensitive_name(name: str, rules: RuleConfig) -> bool:
    low = name.lower()
    return any(k in low for k in rules.sensitive_name_keywords)


def contains_sensitive_path(path: Optional[Any], rules: RuleConfig) -> bool:
    if not isinstance(path, str):
        return False
    low = path.lower()
    return any(k in low for k in rules.sensitive_path_keywords)


# ============================================================
# 【修改位置 3】论文模式分类辅助函数
# ============================================================

def match_call_pattern(name: str, pattern_map: Dict[str, Set[str]]) -> str:
    for pattern, calls in pattern_map.items():
        if is_rule_match(name, calls):
            return pattern
    return ""


def classify_sensitive_name_pattern(name: str) -> str:
    low = name.lower()
    agent_context = {
        "prompt", "user_prompt", "system_prompt", "developer_message", "messages", "message", "chat_history",
        "history", "conversation", "memory", "agent_memory", "long_term_memory", "tool_outputs", "tool_result",
        "observations", "retrieved_docs", "context_chunks", "scratchpad", "trajectory", "intermediate_steps",
        "state", "context",
    }
    dynamic_dispatch = {
        "module_name", "function_name", "func_name", "class_name", "handler_name", "plugin_name", "tool_name", "method_name",
    }
    if any(k in low for k in agent_context):
        return "P3_Agent_Context_Source"
    if any(k in low for k in dynamic_dispatch):
        return "D8_Dynamic_Import_Reflection"
    return "E2_Credential_Harvesting"


def classify_sensitive_path_pattern(path: Optional[Any]) -> str:
    if not isinstance(path, str):
        return "E3_File_System_Enumeration"
    low = path.lower()
    credential_markers = {
        ".env", ".npmrc", ".pypirc", "id_rsa", "id_dsa", "id_ed25519", ".ssh", ".aws/credentials",
        ".aws/config", ".gnupg", "credentials", "credential", "secret", "secrets", "token", "apikey",
        "api_key", "passwd", "shadow", "cookies.sqlite", "login data", ".git-credentials",
    }
    if any(k in low for k in credential_markers):
        return "PE3_Credential_File_Access"
    return "E3_File_System_Enumeration"


def source_label_from_pattern(pattern: str, fallback: str) -> str:
    return pattern if pattern else fallback


def sink_label_from_pattern(pattern: str, fallback: str) -> str:
    return pattern if pattern else fallback


# ============================================================
# 4. PDG Builder：AST + CFG + DFG 融合建图
# ============================================================

class DefinitionEnv:
    """变量定义环境。name -> 最近定义节点集合。"""

    def __init__(self, parent: Optional["DefinitionEnv"] = None):
        self.parent = parent
        self.defs: Dict[str, Set[int]] = defaultdict(set)

    def copy(self) -> "DefinitionEnv":
        e = DefinitionEnv(self.parent)
        e.defs = defaultdict(set, {k: set(v) for k, v in self.defs.items()})
        return e

    def get(self, name: str) -> Set[int]:
        if name in self.defs and self.defs[name]:
            return set(self.defs[name])
        if self.parent:
            return self.parent.get(name)
        return set()

    def set(self, name: str, node_id: int) -> None:
        self.defs[name] = {node_id}

    def merge_from(self, other: "DefinitionEnv") -> None:
        for k, v in other.defs.items():
            self.defs[k] |= set(v)


@dataclass
class FunctionInfo:
    name: str
    node: ast.AST
    params: List[str]
    param_ids: Dict[str, int] = field(default_factory=dict)
    return_ids: Set[int] = field(default_factory=set)


class PDGBuilder(ast.NodeVisitor):
    """从 Python AST 构建 PDG。

    建图分三层：
    1. build_ast_edges：遍历 AST，创建所有节点和 AST 父子边。
    2. build_cfg：对语句序列创建 CFG 边。
    3. build_dfg：根据定义-使用关系创建 DFG 边，同时标注 Source/Sink/Sanitizer。
    """

    def __init__(self, file: str, source_code: str, rules: Optional[RuleConfig] = None):
        self.file = file
        self.source_code = source_code
        self.rules = rules or RuleConfig()
        self.tree = ast.parse(source_code, filename=file)
        self.pdg = ProgramDependenceGraph(file=file)
        self.next_id = 1
        self.imports: Dict[str, str] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.current_function: Optional[str] = None

    def build(self) -> ProgramDependenceGraph:
        self.build_ast_edges(self.tree, parent_id=0)
        self.collect_imports_and_functions(self.tree)
        self.build_cfg_for_body(getattr(self.tree, "body", []), label="module")
        env = DefinitionEnv()
        self.build_dfg_for_body(getattr(self.tree, "body", []), env)
        return self.pdg

    # -------------------------
    # AST 层：创建节点与 AST 边
    # -------------------------

    def new_node(self, node: ast.AST) -> int:
        if ast_key(node) in self.pdg.ast_to_id:
            return self.pdg.ast_to_id[ast_key(node)]
        line, col = node_location(node)
        label = self.node_label(node)
        code = self.get_source_segment(node)
        node_id = self.next_id
        self.next_id += 1
        pdg_node = PDGNode(
            id=node_id,
            kind="ast",
            label=label,
            file=self.file,
            line=line,
            col=col,
            code=code,
            ast_type=type(node).__name__,
        )
        self.pdg.ast_to_id[ast_key(node)] = node_id
        self.pdg.add_node(pdg_node)
        return node_id

    def build_ast_edges(self, node: ast.AST, parent_id: int = 0) -> int:
        node_id = self.new_node(node)
        if parent_id:
            self.pdg.add_edge(parent_id, node_id, "ast", "child")
        for child in ast.iter_child_nodes(node):
            self.build_ast_edges(child, node_id)
        return node_id

    def node_label(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return attr_chain(node) or node.attr
        if isinstance(node, ast.Call):
            return call_name(node, self.imports)
        if isinstance(node, ast.FunctionDef):
            return f"def {node.name}"
        if isinstance(node, ast.AsyncFunctionDef):
            return f"async def {node.name}"
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Assign):
            return "assign"
        if isinstance(node, ast.Return):
            return "return"
        return type(node).__name__

    def get_source_segment(self, node: ast.AST) -> str:
        try:
            seg = ast.get_source_segment(self.source_code, node)
            return seg.strip() if seg else ""
        except Exception:
            return ""

    # -------------------------
    # 符号收集：import / function
    # -------------------------

    def collect_imports_and_functions(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = f"{mod}.{alias.name}" if mod else alias.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [a.arg for a in node.args.args]
                info = FunctionInfo(name=node.name, node=node, params=params)
                for a in node.args.args:
                    info.param_ids[a.arg] = self.pdg.ast_to_id.get(ast_key(a), self.new_node(a))
                self.functions[node.name] = info

    # -------------------------
    # CFG 层：语句间控制流
    # -------------------------

    def build_cfg_for_body(self, body: List[ast.stmt], label: str = "") -> Tuple[Optional[int], Optional[int]]:
        if not body:
            return None, None
        first_id = self.pdg.ast_to_id[ast_key(body[0])]
        prev_exits: List[int] = []
        for stmt in body:
            entry, exits = self.build_cfg_stmt(stmt)
            for p in prev_exits:
                if entry:
                    self.pdg.add_edge(p, entry, "cfg", "next")
            prev_exits = exits
        return first_id, prev_exits[-1] if prev_exits else first_id

    def build_cfg_stmt(self, stmt: ast.stmt) -> Tuple[int, List[int]]:
        sid = self.pdg.ast_to_id[ast_key(stmt)]
        if isinstance(stmt, ast.If):
            test_id = self.pdg.ast_to_id[ast_key(stmt.test)]
            self.pdg.add_edge(sid, test_id, "cfg", "condition")
            body_entry, body_exit = self.build_cfg_for_body(stmt.body, "if-body")
            else_entry, else_exit = self.build_cfg_for_body(stmt.orelse, "if-else")
            if body_entry:
                self.pdg.add_edge(test_id, body_entry, "cfg", "true")
            if else_entry:
                self.pdg.add_edge(test_id, else_entry, "cfg", "false")
            exits = []
            if body_exit:
                exits.append(body_exit)
            if else_exit:
                exits.append(else_exit)
            if not exits:
                exits = [test_id]
            return sid, exits

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            iter_id = self.pdg.ast_to_id[ast_key(stmt.iter)]
            self.pdg.add_edge(sid, iter_id, "cfg", "iter")
            body_entry, body_exit = self.build_cfg_for_body(stmt.body, "for-body")
            if body_entry:
                self.pdg.add_edge(iter_id, body_entry, "cfg", "loop-true")
            if body_exit:
                self.pdg.add_edge(body_exit, iter_id, "cfg", "loop-back")
            return sid, [iter_id]

        if isinstance(stmt, ast.While):
            test_id = self.pdg.ast_to_id[ast_key(stmt.test)]
            self.pdg.add_edge(sid, test_id, "cfg", "condition")
            body_entry, body_exit = self.build_cfg_for_body(stmt.body, "while-body")
            if body_entry:
                self.pdg.add_edge(test_id, body_entry, "cfg", "true")
            if body_exit:
                self.pdg.add_edge(body_exit, test_id, "cfg", "loop-back")
            return sid, [test_id]

        if isinstance(stmt, ast.Try):
            body_entry, body_exit = self.build_cfg_for_body(stmt.body, "try-body")
            if body_entry:
                self.pdg.add_edge(sid, body_entry, "cfg", "try")
            exits = []
            if body_exit:
                exits.append(body_exit)
            for h in stmt.handlers:
                h_id = self.pdg.ast_to_id[ast_key(h)]
                self.pdg.add_edge(sid, h_id, "cfg", "except")
                h_entry, h_exit = self.build_cfg_for_body(h.body, "except-body")
                if h_entry:
                    self.pdg.add_edge(h_id, h_entry, "cfg", "handler")
                if h_exit:
                    exits.append(h_exit)
            final_entry, final_exit = self.build_cfg_for_body(stmt.finalbody, "finally")
            if final_entry:
                for e in exits or [sid]:
                    self.pdg.add_edge(e, final_entry, "cfg", "finally")
                exits = [final_exit] if final_exit else [final_entry]
            return sid, exits or [sid]

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # 函数/类定义作为普通语句存在；函数体内部单独构建 CFG。
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.build_cfg_for_body(stmt.body, f"function:{stmt.name}")
            return sid, [sid]

        return sid, [sid]

    # -------------------------
    # DFG 层：定义-使用数据流 + Source/Sink 标注
    # -------------------------

    def build_dfg_for_body(self, body: List[ast.stmt], env: DefinitionEnv) -> None:
        for stmt in body:
            self.build_dfg_stmt(stmt, env)

    def build_dfg_stmt(self, stmt: ast.stmt, env: DefinitionEnv) -> None:
        sid = self.pdg.ast_to_id[ast_key(stmt)]

        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return

        if isinstance(stmt, ast.FunctionDef):
            info = self.functions.get(stmt.name)
            if info:
                env.set(stmt.name, sid)
                fn_env = DefinitionEnv(parent=env)
                for p in stmt.args.args:
                    pid = self.pdg.ast_to_id.get(ast_key(p), sid)
                    fn_env.set(p.arg, pid)
                    # 【修改位置 4.1】参数名启发式 Source，例如 password/token/prompt/context。
                    if contains_sensitive_name(p.arg, self.rules):
                        self.pdg.mark_source(pid, f"sensitive-param:{p.arg}")
                        self.pdg.nodes[pid].meta.setdefault("source_pattern", classify_sensitive_name_pattern(p.arg))
                old = self.current_function
                self.current_function = stmt.name
                self.build_dfg_for_body(stmt.body, fn_env)
                self.current_function = old
            return

        if isinstance(stmt, ast.AsyncFunctionDef):
            self.build_dfg_stmt(ast.FunctionDef(
                name=stmt.name,
                args=stmt.args,
                body=stmt.body,
                decorator_list=stmt.decorator_list,
                returns=stmt.returns,
                type_comment=getattr(stmt, "type_comment", None),
            ), env)
            return

        if isinstance(stmt, ast.Assign):
            value_id = self.expr_representative_id(stmt.value)
            self.connect_expr_uses_to_node(stmt.value, value_id, env, "use")
            self.detect_expr_source_sink(stmt.value, env)
            for target in stmt.targets:
                for name in target_names(target):
                    target_id = self.pdg.ast_to_id.get(ast_key(target), sid)
                    # value 表达式到赋值目标，表示值流入变量定义点。
                    self.pdg.add_edge(value_id, target_id, "dfg", f"assign:{name}")
                    env.set(name, target_id)
                    # 【修改位置 4.2】赋值目标变量名启发式 Source，例如 api_key/chat_history/module_name。
                    if contains_sensitive_name(name, self.rules):
                        self.pdg.mark_source(target_id, f"sensitive-name:{name}")
                        self.pdg.nodes[target_id].meta.setdefault("source_pattern", classify_sensitive_name_pattern(name))
            return

        if isinstance(stmt, ast.AnnAssign):
            if stmt.value:
                value_id = self.expr_representative_id(stmt.value)
                self.connect_expr_uses_to_node(stmt.value, value_id, env, "use")
                self.detect_expr_source_sink(stmt.value, env)
                for name in target_names(stmt.target):
                    target_id = self.pdg.ast_to_id.get(ast_key(stmt.target), sid)
                    self.pdg.add_edge(value_id, target_id, "dfg", f"assign:{name}")
                    env.set(name, target_id)
            return

        if isinstance(stmt, ast.AugAssign):
            target_id = self.pdg.ast_to_id.get(ast_key(stmt.target), sid)
            for name in target_names(stmt.target):
                for d in env.get(name):
                    self.pdg.add_edge(d, target_id, "dfg", f"aug-old:{name}")
                env.set(name, target_id)
            self.connect_expr_uses_to_node(stmt.value, target_id, env, "aug-value")
            self.detect_expr_source_sink(stmt.value, env)
            return

        if isinstance(stmt, ast.Expr):
            expr_id = self.expr_representative_id(stmt.value)
            self.connect_expr_uses_to_node(stmt.value, expr_id, env, "expr-use")
            self.detect_expr_source_sink(stmt.value, env)
            return

        if isinstance(stmt, ast.Return):
            if stmt.value:
                self.connect_expr_uses_to_node(stmt.value, sid, env, "return-use")
                self.detect_expr_source_sink(stmt.value, env)
                if self.current_function and self.current_function in self.functions:
                    self.functions[self.current_function].return_ids.add(sid)
            return

        if isinstance(stmt, ast.If):
            self.connect_expr_uses_to_node(stmt.test, self.pdg.ast_to_id[ast_key(stmt.test)], env, "if-test")
            self.detect_expr_source_sink(stmt.test, env)
            env_body = env.copy()
            env_else = env.copy()
            self.build_dfg_for_body(stmt.body, env_body)
            self.build_dfg_for_body(stmt.orelse, env_else)
            env.merge_from(env_body)
            env.merge_from(env_else)
            return

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            iter_id = self.expr_representative_id(stmt.iter)
            self.connect_expr_uses_to_node(stmt.iter, iter_id, env, "iter-use")
            self.detect_expr_source_sink(stmt.iter, env)
            for name in target_names(stmt.target):
                tid = self.pdg.ast_to_id.get(ast_key(stmt.target), sid)
                self.pdg.add_edge(iter_id, tid, "dfg", f"for-target:{name}")
                env.set(name, tid)
            self.build_dfg_for_body(stmt.body, env)
            self.build_dfg_for_body(stmt.orelse, env)
            return

        if isinstance(stmt, ast.While):
            self.connect_expr_uses_to_node(stmt.test, self.pdg.ast_to_id[ast_key(stmt.test)], env, "while-test")
            self.detect_expr_source_sink(stmt.test, env)
            self.build_dfg_for_body(stmt.body, env)
            self.build_dfg_for_body(stmt.orelse, env)
            return

        if isinstance(stmt, ast.With):
            for item in stmt.items:
                ctx_id = self.expr_representative_id(item.context_expr)
                self.connect_expr_uses_to_node(item.context_expr, ctx_id, env, "with-context")
                self.detect_expr_source_sink(item.context_expr, env)
                if item.optional_vars:
                    for name in target_names(item.optional_vars):
                        vid = self.pdg.ast_to_id.get(ast_key(item.optional_vars), sid)
                        self.pdg.add_edge(ctx_id, vid, "dfg", f"with-as:{name}")
                        env.set(name, vid)
            self.build_dfg_for_body(stmt.body, env)
            return

        if isinstance(stmt, ast.Try):
            self.build_dfg_for_body(stmt.body, env)
            for h in stmt.handlers:
                self.build_dfg_for_body(h.body, env)
            self.build_dfg_for_body(stmt.orelse, env)
            self.build_dfg_for_body(stmt.finalbody, env)
            return

        # 默认保守处理：把语句内部表达式的 use 连到语句节点。
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                self.connect_expr_uses_to_node(child, sid, env, "stmt-use")
                self.detect_expr_source_sink(child, env)

    def expr_representative_id(self, expr: ast.AST) -> int:
        """表达式代表节点：Call 用自身，其他表达式用自身 AST 节点。"""
        return self.pdg.ast_to_id.get(ast_key(expr), self.new_node(expr))

    def connect_expr_uses_to_node(self, expr: ast.AST, dst_id: int, env: DefinitionEnv, label: str) -> None:
        """把表达式中使用到的变量最近定义点连到 dst_id。"""
        for name in used_names(expr):
            for def_id in env.get(name):
                self.pdg.add_edge(def_id, dst_id, "dfg", f"{label}:{name}")

    def detect_expr_source_sink(self, expr: ast.AST, env: DefinitionEnv) -> None:
        """遍历表达式内部调用，识别 Source / Sink / Sanitizer，并建立调用相关 DFG。"""
        for node in ast.walk(expr):
            if not isinstance(node, ast.Call):
                continue
            cid = self.pdg.ast_to_id[ast_key(node)]
            name = call_name(node, self.imports)

            # 实参变量定义点 -> call 节点
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                self.connect_expr_uses_to_node(arg, cid, env, "arg")

            # 【修改位置 5.1】Source 标注：匹配论文 E2/SC2/SC3/P3 等 Source 模式
            if is_rule_match(name, self.rules.source_calls):
                self.pdg.mark_source(cid, f"source-call:{name}")
                pattern = match_call_pattern(name, self.rules.source_pattern_calls)
                self.pdg.nodes[cid].meta.setdefault("source_pattern", source_label_from_pattern(pattern, "UNTRUSTED_INPUT"))

            # 【修改位置 5.2】File Source 标注：敏感路径读取/枚举，对应 PE3 或 E3
            if is_rule_match(name, self.rules.file_read_calls):
                first = node.args[0] if node.args else None
                path = literal_value(first)
                if contains_sensitive_path(path, self.rules):
                    self.pdg.mark_source(cid, f"sensitive-file:{path}")
                    self.pdg.nodes[cid].meta.setdefault("source_pattern", classify_sensitive_path_pattern(path))

            # Sanitizer 标注
            if is_rule_match(name, self.rules.sanitizer_calls):
                self.pdg.mark_sanitizer(cid, f"sanitizer:{name}")

            # 【修改位置 5.3】Sink 标注：匹配论文 E1/P3/SC1/SC2/PE2/D8 等 Sink 模式
            if is_rule_match(name, self.rules.sink_calls):
                self.pdg.mark_sink(cid, f"sink-call:{name}")
                pattern = match_call_pattern(name, self.rules.sink_pattern_calls)
                self.pdg.nodes[cid].meta.setdefault("sink_pattern", sink_label_from_pattern(pattern, "UNKNOWN_SINK"))

            # 【修改位置 5.4】P3 复合模式弱识别：exec/eval 的参数中出现 requests/urllib/httpx/socket 等外传关键词
            if name in {"eval", "exec"} and node.args:
                arg_text = self.get_source_segment(node.args[0]).lower()
                if any(k in arg_text for k in ["requests", "urllib", "httpx", "socket", "http://", "https://"]):
                    self.pdg.mark_sink(cid, f"sink-call:{name}")
                    self.pdg.nodes[cid].meta.setdefault("sink_pattern", "P3_Context_or_Data_Exposure")

            # 轻量级跨函数边：call args -> params, returns -> call
            simple_name = name.split(".")[-1]
            if simple_name in self.functions:
                info = self.functions[simple_name]
                for idx, arg in enumerate(node.args):
                    if idx < len(info.params):
                        param = info.params[idx]
                        param_id = info.param_ids.get(param)
                        if param_id:
                            # 参数表达式使用点流向函数形参
                            arg_id = self.expr_representative_id(arg)
                            self.connect_expr_uses_to_node(arg, arg_id, env, "arg-to-param")
                            self.pdg.add_edge(arg_id, param_id, "call", f"arg->{param}")
                for ret_id in info.return_ids:
                    self.pdg.add_edge(ret_id, cid, "return", f"return->{simple_name}")


# ============================================================
# 5. PDG 污点传播：在 PDG 上做可达性分析
# ============================================================

class PDGTaintAnalyzer:
    """基于 PDG 的静态污点传播。

    默认只沿数据相关边传播：dfg、call、return、sanitize。
    cfg/ast 边主要用于解释结构，不默认作为污点传播边，避免过度污染。
    """

    def __init__(self, pdg: ProgramDependenceGraph, propagation_edges: Optional[Set[str]] = None):
        self.pdg = pdg
        self.propagation_edges = propagation_edges or {"dfg", "call", "return", "sanitize"}

    def analyze(self) -> List[Finding]:
        findings: List[Finding] = []
        for source_id in sorted(self.pdg.source_nodes):
            findings.extend(self.propagate_from_source(source_id))
        return self.deduplicate(findings)

    def propagate_from_source(self, source_id: int) -> List[Finding]:
        q: deque[Tuple[int, List[int]]] = deque()
        q.append((source_id, [source_id]))
        visited: Set[int] = {source_id}
        findings: List[Finding] = []

        while q:
            nid, path = q.popleft()
            if nid in self.pdg.sink_nodes and nid != source_id:
                findings.append(self.make_finding(source_id, nid, path))
                continue

            for edge in self.pdg.out_edges.get(nid, []):
                if edge.edge_type not in self.propagation_edges:
                    continue

                # 如果进入 sanitizer 节点，默认停止继续传播。需要“净化后仍检测”可改这里。
                if edge.dst in self.pdg.sanitizer_nodes:
                    continue

                if edge.dst not in visited:
                    visited.add(edge.dst)
                    q.append((edge.dst, path + [edge.dst]))
        return findings

    def make_finding(self, source_id: int, sink_id: int, path: List[int]) -> Finding:
        """【修改位置 6】生成 Finding 时补充论文模式、严重级别、Kill Chain 和攻击链标签。"""
        src = self.pdg.nodes[source_id]
        sink = self.pdg.nodes[sink_id]
        source_label = src.meta.get("source", src.label)
        sink_label = sink.meta.get("sink", sink.label)
        source_pattern = src.meta.get("source_pattern", "")
        sink_pattern = sink.meta.get("sink_pattern", "")
        defect_type, severity, kill_chain_phase, attack_chain = self.classify_finding(
            str(source_label), str(sink_label), str(source_pattern), str(sink_pattern)
        )
        return Finding(
            file=self.pdg.file,
            source_id=source_id,
            sink_id=sink_id,
            source=source_label,
            sink=sink_label,
            line=sink.line,
            col=sink.col,
            code=sink.code,
            path=path,
            path_labels=[self.format_path_node(i) for i in path],
            severity=severity,
            kind="pdg-static-taint-flow",
            defect_type=defect_type,
            source_pattern=source_pattern,
            sink_pattern=sink_pattern,
            kill_chain_phase=kill_chain_phase,
            attack_chain=attack_chain,
        )

    def format_path_node(self, node_id: int) -> str:
        n = self.pdg.nodes[node_id]
        tag = ""
        if "source" in n.tags:
            tag = "[SOURCE]"
        elif "sink" in n.tags:
            tag = "[SINK]"
        elif "sanitizer" in n.tags:
            tag = "[SANITIZER]"
        return f"#{n.id}{tag} {n.ast_type} {n.label} L{n.line}"

    def classify_finding(
        self,
        source_label: str,
        sink_label: str,
        source_pattern: str,
        sink_pattern: str,
    ) -> Tuple[str, str, str, str]:
        """【修改位置 7】根据 Source/Sink 组合输出论文式缺陷类型。"""
        source_low = source_label.lower()
        sink_low = sink_label.lower()

        if source_pattern == "SC2_Remote_Content_Source" and (
            sink_pattern in {"D5_Dynamic_Code_Execution", "SC1_Command_Injection", "SC2_Remote_Script_Execution"}
            or any(x in sink_low for x in ["exec", "eval", "subprocess", "os.system", "pip"])
        ):
            return "SC2_Remote_Script_Execution", "CRITICAL", "Execution", "Remote Content -> Code/Command Execution"

        if source_pattern == "SC3_Obfuscated_Decode_Source" and (
            sink_pattern in {"D5_Dynamic_Code_Execution", "SC1_Command_Injection"}
            or any(x in sink_low for x in ["exec", "eval", "subprocess", "os.system"])
        ):
            return "SC3_Obfuscated_Code_Execution", "CRITICAL", "Defense Evasion -> Execution", "Obfuscated Decode -> Code/Command Execution"

        if source_pattern in {"E2_Credential_Harvesting", "PE3_Credential_File_Access"} and sink_pattern == "E1_External_Transmission":
            return "E2_PE3_to_E1_Credential_Exfiltration", "CRITICAL", "Credential Access -> Exfiltration", "Credential/File Source -> External Transmission"

        if source_pattern == "P3_Agent_Context_Source" and sink_pattern in {"E1_External_Transmission", "P3_Context_or_Data_Exposure"}:
            return "P3_Agent_Context_Leakage", "HIGH", "Exfiltration", "Agent Context -> External/Local Exposure"

        if sink_pattern == "D5_Dynamic_Code_Execution":
            return "D5_Untrusted_Input_to_Dynamic_Code_Execution", "HIGH", "Execution", "Source -> eval/exec/compile"

        if sink_pattern == "SC1_Command_Injection":
            return "SC1_Command_Injection", "HIGH", "Execution", "Source -> System Command Execution"

        if sink_pattern == "D8_Dynamic_Import_Reflection":
            return "D8_Dynamic_Import_or_Reflection_Call", "HIGH", "Execution", "Source -> Dynamic Import/Reflection"

        if sink_pattern == "PE2_Privilege_Escalation":
            return "PE2_Privilege_Escalation", "MEDIUM", "Impact", "Source -> Privilege-Sensitive Operation"

        if sink_pattern == "E1_External_Transmission":
            sev = "HIGH"
            if any(x in source_low for x in ["token", "secret", "password", "credential", "api_key", "private_key"]):
                sev = "CRITICAL"
            return "E1_External_Transmission", sev, "Exfiltration", "Source -> External Transmission"

        if sink_pattern == "P3_Context_or_Data_Exposure":
            return "P3_Context_or_Data_Exposure", "HIGH", "Exfiltration", "Source -> Log/Print/File Exposure"

        return "Generic_Source_to_Sink_Flow", self.severity_for_sink(sink_label), "Unknown", "Source -> Sink"

    def severity_for_sink(self, sink: str) -> str:
        if any(x in sink for x in ["eval", "exec", "subprocess", "os.system", "Popen"]):
            return "HIGH"
        if any(x in sink for x in ["request", "httpx", "urllib", "socket", "sendmail", "openai", "anthropic"]):
            return "HIGH"
        return "MEDIUM"

    def deduplicate(self, findings: List[Finding]) -> List[Finding]:
        seen: Set[Tuple[str, int, int, str, str]] = set()
        out: List[Finding] = []
        for f in findings:
            key = (f.file, f.source_id, f.sink_id, f.source, f.sink)
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out


# ============================================================
# 6. 项目扫描与输出
# ============================================================

def iter_python_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix == ".py":
        yield path
        return
    if path.is_dir():
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", "dist", "build"}
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                if f.endswith(".py"):
                    yield Path(root) / f


def analyze_file(file: Path, propagation_edges: Optional[Set[str]] = None) -> Tuple[ProgramDependenceGraph, List[Finding]]:
    source = file.read_text(encoding="utf-8", errors="replace")
    builder = PDGBuilder(str(file), source)
    pdg = builder.build()
    findings = PDGTaintAnalyzer(pdg, propagation_edges=propagation_edges).analyze()
    return pdg, findings


# def analyze_path(path: str, propagation_edges: Optional[Set[str]] = None) -> Tuple[List[ProgramDependenceGraph], List[Finding]]:
#     pdgs: List[ProgramDependenceGraph] = []
#     findings: List[Finding] = []
#     for file in iter_python_files(Path(path)):
#         try:
#             pdg, fs = analyze_file(file, propagation_edges=propagation_edges)
#             pdgs.append(pdg)
#             findings.extend(fs)
#         except SyntaxError as e:
#             print(f"[WARN] SyntaxError: {file}: {e}", file=sys.stderr)
#         except Exception as e:
#             print(f"[WARN] Failed to analyze {file}: {e}", file=sys.stderr)
#     return pdgs, findings

def analyze_path(
    path: str,
    propagation_edges: Optional[Set[str]] = None,
    per_file_result_name: str = "taint_result.json",
    per_file_pdg_result_name: str = "pdg_result.json"
) -> Tuple[List[ProgramDependenceGraph], List[Finding]]:
    """
    分析目标路径下所有 Python 文件。

    每分析一个源代码文件，就在该源代码文件所在目录生成：
    - taint_result.json：该源文件的污点分析结果
    - pdg_result.json：该源文件的 PDG 节点与边结果

    同时仍然返回所有 pdg 和 findings，方便全局汇总。
    """
    pdgs: List[ProgramDependenceGraph] = []
    all_findings: List[Finding] = []

    for file in iter_python_files(Path(path)):
        try:
            print(f"[analyze] 正在分析：{file}")

            pdg, file_findings = analyze_file(file, propagation_edges=propagation_edges)

            pdgs.append(pdg)
            all_findings.extend(file_findings)

            # 关键修改：每个源代码文件生成对应 taint_result.json
            write_taint_result_for_source_file(
                source_file=file,
                findings=file_findings,
                result_name=per_file_result_name
            )

            # 关键修改：每个源代码文件生成对应 pdg_result.json
            write_pdg_result_for_source_file(
                source_file=file,
                pdg=pdg,
                result_name=per_file_pdg_result_name
            )

        except SyntaxError as e:
            print(f"[WARN] SyntaxError: {file}: {e}", file=sys.stderr)

            # 即使语法错误，也在源代码文件所在目录生成错误结果，方便定位失败文件。
            write_error_result_for_source_file(
                source_file=file,
                result_name=per_file_result_name,
                result_kind="taint",
                error=f"SyntaxError: {e}"
            )
            write_error_result_for_source_file(
                source_file=file,
                result_name=per_file_pdg_result_name,
                result_kind="pdg",
                error=f"SyntaxError: {e}"
            )

        except Exception as e:
            print(f"[WARN] Failed to analyze {file}: {e}", file=sys.stderr)

            # 其他异常也写入当前文件目录下的 taint_result.json 和 pdg_result.json。
            error_text = f"{type(e).__name__}: {e}"
            write_error_result_for_source_file(
                source_file=file,
                result_name=per_file_result_name,
                result_kind="taint",
                error=error_text
            )
            write_error_result_for_source_file(
                source_file=file,
                result_name=per_file_pdg_result_name,
                result_kind="pdg",
                error=error_text
            )

    return pdgs, all_findings


def findings_to_json(findings: List[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], indent=2, ensure_ascii=False)


def print_human_report(findings: List[Finding]) -> None:
    if not findings:
        print("未发现 Source → Sink 污点流。")
        return

    print(f"发现 {len(findings)} 条基于 PDG 的潜在污点流：\n")
    for idx, f in enumerate(findings, 1):
        # 【修改位置 8】人类可读报告增加论文模式、严重级别、Kill Chain 信息。
        print(f"[{idx}] {f.severity} {f.kind}")
        print(f"  缺陷类型 : {f.defect_type}")
        print(f"  攻击阶段 : {f.kill_chain_phase}")
        print(f"  攻击链   : {f.attack_chain}")
        print(f"  文件位置 : {f.file}:{f.line}:{f.col}")
        print(f"  Source   : {f.source}  节点 #{f.source_id}  模式={f.source_pattern}")
        print(f"  Sink     : {f.sink}  节点 #{f.sink_id}  模式={f.sink_pattern}")
        if f.code:
            print(f"  代码片段 : {f.code}")
        print("  PDG Path:")
        for p in f.path_labels:
            print(f"    - {p}")
        print()


def write_dot_for_first_pdg(pdgs: List[ProgramDependenceGraph], dot_path: str, edge_types: Optional[Set[str]]) -> None:
    if not pdgs:
        return
    Path(dot_path).write_text(pdgs[0].to_dot(edge_types=edge_types), encoding="utf-8")


def write_taint_result_for_source_file(
    source_file: Path,
    findings: List[Finding],
    result_name: str = "taint_result.json"
) -> None:
    """
    在每个源代码文件所在目录生成 taint_result.json。

    如果同一个目录下有多个 .py 文件，为避免互相覆盖，
    taint_result.json 内部使用 files 字段区分不同源代码文件。
    """
    result_path = source_file.parent / result_name

    current_payload = {
        "result_file": str(result_path),
        "files": {}
    }

    # 如果该目录下已经有 taint_result.json，说明可能之前分析过同目录其他 .py 文件。
    # 此时读取旧结果并合并，避免覆盖。
    if result_path.exists():
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "files" in old:
                current_payload = old
        except Exception:
            pass

    current_payload["files"][source_file.name] = {
        "source_file": str(source_file),
        "finding_count": len(findings),
        "findings": [asdict(f) for f in findings]
    }

    result_path.write_text(
        json.dumps(current_payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # print(f"[per-file-json] 已生成：{result_path}")


def pdg_to_json_dict(pdg: ProgramDependenceGraph) -> Dict[str, Any]:
    """
    将 ProgramDependenceGraph 转成可 JSON 序列化的字典。

    注意：PDGNode.tags 是 set，不能直接 json.dumps(asdict(node))，
    因此这里显式转成排序后的 list。
    """
    return {
        "source_file": pdg.file,
        "node_count": len(pdg.nodes),
        "edge_count": len(pdg.edges),
        "source_nodes": sorted(pdg.source_nodes),
        "sink_nodes": sorted(pdg.sink_nodes),
        "sanitizer_nodes": sorted(pdg.sanitizer_nodes),
        "nodes": {
            str(node_id): {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "file": node.file,
                "line": node.line,
                "col": node.col,
                "code": node.code,
                "ast_type": node.ast_type,
                "tags": sorted(node.tags),
                "meta": node.meta,
            }
            for node_id, node in sorted(pdg.nodes.items())
        },
        "edges": [asdict(edge) for edge in pdg.edges],
    }


def write_pdg_result_for_source_file(
    source_file: Path,
    pdg: ProgramDependenceGraph,
    result_name: str = "pdg_result.json"
) -> None:
    """
    在每个源代码文件所在目录生成 pdg_result.json。

    如果同一个目录下有多个 .py 文件，为避免互相覆盖，
    pdg_result.json 内部使用 files 字段区分不同源代码文件。
    """
    result_path = source_file.parent / result_name

    current_payload = {
        "result_file": str(result_path),
        "files": {}
    }

    if result_path.exists():
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "files" in old:
                current_payload = old
        except Exception:
            pass

    current_payload["files"][source_file.name] = pdg_to_json_dict(pdg)

    result_path.write_text(
        json.dumps(current_payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def write_error_result_for_source_file(
    source_file: Path,
    result_name: str,
    result_kind: str,
    error: str
) -> None:
    """
    分析失败时，也在源代码文件所在目录生成对应结果文件，
    使失败文件和失败原因与源码目录保持一一对应。
    """
    result_path = source_file.parent / result_name

    current_payload = {
        "result_file": str(result_path),
        "files": {}
    }

    if result_path.exists():
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and "files" in old:
                current_payload = old
        except Exception:
            pass

    if result_kind == "pdg":
        current_payload["files"][source_file.name] = {
            "source_file": str(source_file),
            "node_count": 0,
            "edge_count": 0,
            "source_nodes": [],
            "sink_nodes": [],
            "sanitizer_nodes": [],
            "nodes": {},
            "edges": [],
            "error": error,
        }
    else:
        current_payload["files"][source_file.name] = {
            "source_file": str(source_file),
            "finding_count": 0,
            "findings": [],
            "error": error,
        }

    result_path.write_text(
        json.dumps(current_payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="基于 AST+CFG+DFG 构建 PDG 的 Python 静态污点分析器")
    parser.add_argument("target", nargs="?", default=r"/root/BATaint/datasets/repo", help="待分析的 Python 文件或项目目录")
    parser.add_argument("--json", dest="json_path", default=r"/root/BATaint/datasets/repo/pdg_taint_result.json", help="将 findings 保存为 JSON")
    parser.add_argument(
        "--per-file-result-name",
        dest="per_file_result_name",
        default="taint_result.json",
        help="每个源代码文件所在目录生成的污点分析结果文件名，默认 taint_result.json。"
    )
    parser.add_argument(
        "--per-file-pdg-result-name",
        dest="per_file_pdg_result_name",
        default="pdg_result.json",
        help="每个源代码文件所在目录生成的 PDG 结果文件名，默认 pdg_result.json。"
    )
    parser.add_argument(
        "--edge-types",
        nargs="*",
        default=["ast", "cfg", "dfg", "call", "return", "sanitize"],
        help="DOT 输出或传播使用的边类型，例如 ast cfg dfg call return sanitize。默认 DOT 输出全部边，污点传播默认 dfg/call/return/sanitize。",
    )
    parser.add_argument(
        "--propagate-edge-types",
        nargs="*",
        default=["dfg", "call", "return", "sanitize"],
        help="污点传播使用的边类型，默认 dfg call return sanitize。一般不建议加入 ast/cfg，容易过度污染。",
    )
    args = parser.parse_args(argv)

    dot_edge_types = set(args.edge_types) if args.edge_types else None
    propagation_edges = set(args.propagate_edge_types) if args.propagate_edge_types else None

    # pdgs, findings = analyze_path(args.target, propagation_edges=propagation_edges)
    pdgs, findings = analyze_path(
        args.target,
        propagation_edges=propagation_edges,
        per_file_result_name=args.per_file_result_name,
        per_file_pdg_result_name=args.per_file_pdg_result_name
    )
    print_human_report(findings)

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(findings_to_json(findings), encoding="utf-8")
        print(f"JSON 结果已保存：{args.json_path}")

    # if args.dot_path:
    #     write_dot_for_first_pdg(pdgs, args.dot_path, dot_edge_types)
    #     print(f"PDG DOT 已保存：{args.dot_path}")
    #     # print("可使用 graphviz 渲染：dot -Tpng pdg.dot -o pdg.png")

    return 1 if findings else 0


if __name__ == "__main__":
    '''
        生成基于PDG的污点分析报告taint_result.json，
        报告包括：source到sink的传播路径，路径条数findings，PDG的node及节点传递路径
    '''

    raise SystemExit(main())

