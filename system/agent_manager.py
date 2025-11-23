import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

class Agent:
    def __init__(self, agent_id: str, name: str, config: dict, llm_engine):
        self.id = agent_id
        self.name = name
        self.config = config
        self.llm_engine = llm_engine

        self.memory = []
        self.context = {}
        self.tools = []

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        base_prompt = f"""You are {self.name}, an AI agent.

Your capabilities:
- Multilingual communication (English, Serbian, Bosnian, Croatian)
- Task execution and problem solving
- Memory retention across conversations
- Tool usage when available

Current context: {json.dumps(self.context, indent=2)}

Respond naturally and helpfully to user requests."""

        return base_prompt

    def process(self, user_input: str) -> str:
        self.memory.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        messages = [
            {'role': 'system', 'content': self.system_prompt}
        ]

        recent_memory = self.memory[-10:]
        for msg in recent_memory:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

        response = self.llm_engine.chat(messages)

        self.memory.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })

        return response

    def set_context(self, key: str, value: Any):
        self.context[key] = value
        self.system_prompt = self._build_system_prompt()

    def get_context(self, key: str) -> Any:
        return self.context.get(key)

    def clear_memory(self):
        self.memory = []

    def add_tool(self, tool_name: str, tool_function):
        self.tools.append({
            'name': tool_name,
            'function': tool_function
        })

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'config': self.config,
            'memory_size': len(self.memory),
            'context': self.context,
            'tools': [t['name'] for t in self.tools]
        }


class AgentManager:
    def __init__(self, config, llm_engine):
        self.config = config
        self.llm_engine = llm_engine

        self.base_dir = Path(config['system']['install_dir']).expanduser()
        self.agents_dir = self.base_dir / "agents"
        self.agents_dir.mkdir(exist_ok=True)

        self.agents: Dict[str, Agent] = {}

        self.memory_enabled = config['agents'].get('memory_enabled', True)
        self.max_agents = config['agents'].get('max_concurrent', 5)

        if self.memory_enabled:
            self._init_vector_store()

    def _init_vector_store(self):
        chroma_dir = self.base_dir / "chroma"
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        try:
            self.memory_collection = self.chroma_client.get_collection("agent_memory")
        except:
            self.memory_collection = self.chroma_client.create_collection("agent_memory")

    def create_agent(self, name: str, config: Optional[dict] = None) -> Agent:
        if len(self.agents) >= self.max_agents:
            oldest_id = min(self.agents.keys(),
                          key=lambda k: self.agents[k].memory[0]['timestamp']
                          if self.agents[k].memory else datetime.now().isoformat())
            self.delete_agent(oldest_id)

        agent_id = str(uuid.uuid4())
        agent_config = config or {}

        agent = Agent(agent_id, name, agent_config, self.llm_engine)
        self.agents[agent_id] = agent

        self._save_agent(agent)

        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        if agent_id in self.agents:
            return self.agents[agent_id]

        loaded_agent = self._load_agent(agent_id)
        if loaded_agent:
            self.agents[agent_id] = loaded_agent
            return loaded_agent

        return None

    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        for agent in self.agents.values():
            if agent.name == name:
                return agent
        return None

    def run_agent(self, agent_id: str, task: str) -> str:
        agent = self.get_agent(agent_id)
        if not agent:
            return f"Agent {agent_id} not found"

        response = agent.process(task)

        if self.memory_enabled:
            self._store_memory(agent_id, task, response)

        self._save_agent(agent)

        return response

    def list_agents(self) -> List[str]:
        return [f"{agent.name} ({agent.id})" for agent in self.agents.values()]

    def delete_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            del self.agents[agent_id]

        agent_file = self.agents_dir / f"{agent_id}.json"
        if agent_file.exists():
            agent_file.unlink()
            return True

        return False

    def _save_agent(self, agent: Agent):
        agent_file = self.agents_dir / f"{agent.id}.json"

        data = {
            'id': agent.id,
            'name': agent.name,
            'config': agent.config,
            'context': agent.context,
            'memory': agent.memory[-100:],
            'updated': datetime.now().isoformat()
        }

        with open(agent_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_agent(self, agent_id: str) -> Optional[Agent]:
        agent_file = self.agents_dir / f"{agent_id}.json"

        if not agent_file.exists():
            return None

        with open(agent_file, 'r') as f:
            data = json.load(f)

        agent = Agent(
            data['id'],
            data['name'],
            data.get('config', {}),
            self.llm_engine
        )

        agent.context = data.get('context', {})
        agent.memory = data.get('memory', [])

        return agent

    def _store_memory(self, agent_id: str, query: str, response: str):
        embedding = self.embedding_model.encode(query).tolist()

        memory_id = str(uuid.uuid4())

        self.memory_collection.add(
            embeddings=[embedding],
            documents=[response],
            metadatas=[{
                'agent_id': agent_id,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }],
            ids=[memory_id]
        )

    def search_memory(self, agent_id: str, query: str, limit: int = 5) -> List[dict]:
        if not self.memory_enabled:
            return []

        embedding = self.embedding_model.encode(query).tolist()

        results = self.memory_collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={'agent_id': agent_id}
        )

        memories = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                memories.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })

        return memories

    def create_specialized_agent(self, agent_type: str) -> Agent:
        configs = {
            'translator': {
                'name': 'Translator',
                'context': {
                    'specialty': 'Translation',
                    'languages': ['en', 'sr', 'bs', 'hr']
                }
            },
            'coder': {
                'name': 'Coding Assistant',
                'context': {
                    'specialty': 'Software Development',
                    'languages': ['python', 'javascript', 'rust', 'go']
                }
            },
            'researcher': {
                'name': 'Research Assistant',
                'context': {
                    'specialty': 'Research and Analysis'
                }
            },
            'writer': {
                'name': 'Content Writer',
                'context': {
                    'specialty': 'Content Creation'
                }
            }
        }

        config = configs.get(agent_type, configs['researcher'])
        agent = self.create_agent(config['name'], config)

        for key, value in config.get('context', {}).items():
            agent.set_context(key, value)

        return agent

    def get_all_agent_stats(self) -> dict:
        stats = {
            'total_agents': len(self.agents),
            'max_agents': self.max_agents,
            'agents': []
        }

        for agent in self.agents.values():
            stats['agents'].append(agent.to_dict())

        return stats
