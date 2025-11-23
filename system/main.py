import os
import sys
import argparse
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from system.llm_engine import LLMEngine
from system.speech_engine import SpeechEngine
from system.agent_manager import AgentManager
from system.coding_assistant import CodingAssistant
from system.distributed_trainer import DistributedTrainer
from system.api_server import APIServer

console = Console()

class System:
    def __init__(self, config_path=None):
        self.base_dir = Path.home() / "llm_system"
        self.config_path = config_path or self.base_dir / "configs" / "system.yaml"
        self.config = self.load_config()

        console.print(Panel.fit(
            "[bold green]Local LLM System[/bold green]\n"
            "Standalone AI System",
            border_style="green"
        ))

        self.llm_engine = None
        self.speech_engine = None
        self.agent_manager = None
        self.coding_assistant = None
        self.distributed_trainer = None
        self.api_server = None

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize(self):
        console.print("[yellow]Initializing system components...[/yellow]")

        self.llm_engine = LLMEngine(self.config)
        console.print("[green]✓[/green] LLM Engine initialized")

        self.speech_engine = SpeechEngine(self.config)
        console.print("[green]✓[/green] Speech Engine initialized")

        self.agent_manager = AgentManager(self.config, self.llm_engine)
        console.print("[green]✓[/green] Agent Manager initialized")

        self.coding_assistant = CodingAssistant(self.config, self.llm_engine)
        console.print("[green]✓[/green] Coding Assistant initialized")

        if self.config['gpu']['remote']['enabled']:
            self.distributed_trainer = DistributedTrainer(self.config)
            console.print("[green]✓[/green] Distributed Trainer initialized")

        console.print("[bold green]System ready![/bold green]\n")

    def start_interactive(self):
        console.print("[cyan]Starting interactive mode...[/cyan]")
        console.print("Type 'help' for commands, 'exit' to quit\n")

        while True:
            try:
                user_input = input("→ ")

                if user_input.lower() in ['exit', 'quit']:
                    console.print("[yellow]Shutting down...[/yellow]")
                    break

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if user_input.startswith('/agent'):
                    self.handle_agent_command(user_input)
                elif user_input.startswith('/code'):
                    self.handle_code_command(user_input)
                elif user_input.startswith('/voice'):
                    self.handle_voice_command(user_input)
                elif user_input.startswith('/train'):
                    self.handle_train_command(user_input)
                else:
                    response = self.llm_engine.generate(user_input)
                    console.print(f"[green]{response}[/green]\n")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

    def start_server(self, host="0.0.0.0", port=8000):
        console.print(f"[cyan]Starting API server on {host}:{port}...[/cyan]")
        self.api_server = APIServer(
            self.llm_engine,
            self.speech_engine,
            self.agent_manager,
            self.coding_assistant
        )
        self.api_server.start(host, port)

    def handle_agent_command(self, command):
        parts = command.split(maxsplit=2)
        if len(parts) < 2:
            console.print("[red]Usage: /agent <create|list|run> [args][/red]")
            return

        action = parts[1]

        if action == 'create':
            name = parts[2] if len(parts) > 2 else "agent_1"
            agent = self.agent_manager.create_agent(name)
            console.print(f"[green]Agent '{name}' created[/green]")

        elif action == 'list':
            agents = self.agent_manager.list_agents()
            console.print("[cyan]Active agents:[/cyan]")
            for agent in agents:
                console.print(f"  - {agent}")

        elif action == 'run':
            if len(parts) < 3:
                console.print("[red]Usage: /agent run <agent_name> <task>[/red]")
                return
            agent_name, task = parts[2].split(maxsplit=1)
            result = self.agent_manager.run_agent(agent_name, task)
            console.print(f"[green]{result}[/green]")

    def handle_code_command(self, command):
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[red]Usage: /code <description>[/red]")
            return

        task = parts[1]
        result = self.coding_assistant.assist(task)
        console.print(f"[cyan]{result}[/cyan]")

    def handle_voice_command(self, command):
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[red]Usage: /voice <record|speak> [args][/red]")
            return

        action = parts[1]

        if action == 'record':
            console.print("[yellow]Recording... (press Ctrl+C to stop)[/yellow]")
            text = self.speech_engine.record_and_transcribe()
            console.print(f"[cyan]Transcribed: {text}[/cyan]")
            response = self.llm_engine.generate(text)
            console.print(f"[green]{response}[/green]")

        elif action.startswith('speak '):
            text = action[6:]
            self.speech_engine.text_to_speech(text)
            console.print("[green]Audio played[/green]")

    def handle_train_command(self, command):
        if not self.distributed_trainer:
            console.print("[red]Distributed training not enabled[/red]")
            return

        parts = command.split()
        if len(parts) < 2:
            console.print("[red]Usage: /train <start|stop|status>[/red]")
            return

        action = parts[1]

        if action == 'start':
            self.distributed_trainer.start_training()
        elif action == 'stop':
            self.distributed_trainer.stop_training()
        elif action == 'status':
            status = self.distributed_trainer.get_status()
            console.print(f"[cyan]{status}[/cyan]")

    def show_help(self):
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[yellow]Chat:[/yellow]
  Simply type your message to chat with the LLM

[yellow]Agents:[/yellow]
  /agent create <name>           - Create a new agent
  /agent list                    - List active agents
  /agent run <name> <task>       - Run agent with task

[yellow]Coding:[/yellow]
  /code <description>            - Get coding assistance

[yellow]Voice:[/yellow]
  /voice record                  - Record and transcribe
  /voice speak <text>            - Text-to-speech

[yellow]Training:[/yellow]
  /train start                   - Start distributed training
  /train stop                    - Stop training
  /train status                  - Check training status

[yellow]System:[/yellow]
  help                           - Show this help
  exit                           - Exit the system
        """
        console.print(Panel(help_text, border_style="cyan"))

def main():
    parser = argparse.ArgumentParser(description='Local LLM System')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['interactive', 'server'],
                       default='interactive', help='Run mode')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port')

    args = parser.parse_args()

    system = System(config_path=args.config)
    system.initialize()

    if args.mode == 'interactive':
        system.start_interactive()
    elif args.mode == 'server':
        system.start_server(args.host, args.port)

if __name__ == "__main__":
    main()
