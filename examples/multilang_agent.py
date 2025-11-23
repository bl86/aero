#!/usr/bin/env python3

import sys
from pathlib import Path
import yaml
from rich.console import Console

sys.path.insert(0, str(Path.home() / "llm_system"))

from system.agent_manager import AgentManager
from system.llm_engine import LLMEngine

console = Console()

def load_config():
    config_path = Path.home() / "llm_system" / "configs" / "system.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    console.print("[yellow]Initializing Multi-language Agent System...[/yellow]")

    config = load_config()

    llm = LLMEngine(config)
    manager = AgentManager(config, llm)

    console.print("[green]✓ System ready![/green]\n")

    console.print("[cyan]Creating specialized agents...[/cyan]")

    translator = manager.create_specialized_agent('translator')
    console.print(f"[green]✓ Translator agent created: {translator.name}[/green]")

    coder = manager.create_specialized_agent('coder')
    console.print(f"[green]✓ Coder agent created: {coder.name}[/green]")

    researcher = manager.create_specialized_agent('researcher')
    console.print(f"[green]✓ Researcher agent created: {researcher.name}[/green]")

    print("\n" + "="*60)
    print("Multi-language Agent Demonstration")
    print("="*60 + "\n")

    tasks = [
        {
            'agent': translator,
            'language': 'English',
            'task': 'Translate to Serbian: "The future of AI is exciting and full of possibilities."'
        },
        {
            'agent': translator,
            'language': 'Serbian',
            'task': 'Prevedi na hrvatski: "Veštačka inteligencija revolucionira tehnologiju."'
        },
        {
            'agent': coder,
            'language': 'English',
            'task': 'Write a Python function to calculate Fibonacci numbers efficiently.'
        },
        {
            'agent': researcher,
            'language': 'English',
            'task': 'Explain the history and cultural significance of the Balkans region.'
        },
    ]

    for i, task_info in enumerate(tasks, 1):
        agent = task_info['agent']
        language = task_info['language']
        task = task_info['task']

        console.print(f"\n[bold cyan]Task {i} ({language}):[/bold cyan]")
        console.print(f"[dim]Agent: {agent.name}[/dim]")
        console.print(f"[yellow]{task}[/yellow]\n")

        response = manager.run_agent(agent.id, task)

        console.print(f"[green]{response}[/green]")
        console.print("\n" + "-"*60)

    console.print("\n[bold cyan]Agent Statistics:[/bold cyan]")
    stats = manager.get_all_agent_stats()

    console.print(f"Total agents: {stats['total_agents']}")
    for agent_info in stats['agents']:
        console.print(f"\n  - {agent_info['name']}")
        console.print(f"    Memory size: {agent_info['memory_size']}")
        console.print(f"    Context: {agent_info['context']}")

if __name__ == "__main__":
    main()
