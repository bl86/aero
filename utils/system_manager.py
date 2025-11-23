#!/usr/bin/env python3

import subprocess
import psutil
import json
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def get_gpu_info():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'index': parts[0],
                    'name': parts[1],
                    'utilization': parts[2],
                    'memory_used': parts[3],
                    'memory_total': parts[4],
                    'temperature': parts[5]
                })

        return gpus

    except Exception as e:
        return [{'error': str(e)}]

def get_system_info():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory': psutil.virtual_memory()._asdict(),
        'disk': psutil.disk_usage('/')._asdict()
    }

def show_system_status():
    console.print("\n[bold cyan]System Status[/bold cyan]\n")

    sys_info = get_system_info()

    table = Table(title="System Resources")
    table.add_column("Resource", style="cyan")
    table.add_column("Usage", style="yellow")
    table.add_column("Total", style="green")

    table.add_row(
        "CPU",
        f"{sys_info['cpu_percent']}%",
        f"{psutil.cpu_count()} cores"
    )

    mem = sys_info['memory']
    table.add_row(
        "RAM",
        f"{mem['used'] / (1024**3):.1f} GB ({mem['percent']}%)",
        f"{mem['total'] / (1024**3):.1f} GB"
    )

    disk = sys_info['disk']
    table.add_row(
        "Disk",
        f"{disk['used'] / (1024**3):.1f} GB ({disk['percent']}%)",
        f"{disk['total'] / (1024**3):.1f} GB"
    )

    console.print(table)

    gpus = get_gpu_info()

    if gpus and 'error' not in gpus[0]:
        gpu_table = Table(title="GPU Status")
        gpu_table.add_column("Index", style="cyan")
        gpu_table.add_column("Name", style="yellow")
        gpu_table.add_column("Utilization", style="green")
        gpu_table.add_column("Memory", style="blue")
        gpu_table.add_column("Temp", style="red")

        for gpu in gpus:
            gpu_table.add_row(
                gpu['index'],
                gpu['name'],
                f"{gpu['utilization']}%",
                f"{gpu['memory_used']} / {gpu['memory_total']} MB",
                f"{gpu['temperature']}°C"
            )

        console.print("\n")
        console.print(gpu_table)

def check_models():
    models_dir = Path.home() / "llm_system" / "models" / "llm"

    if not models_dir.exists():
        console.print("[red]Models directory not found![/red]")
        return

    models = list(models_dir.iterdir())

    table = Table(title="Installed Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Size", style="yellow")

    for model in models:
        if model.is_dir():
            size = sum(f.stat().st_size for f in model.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            table.add_row(model.name, f"{size_gb:.2f} GB")

    console.print("\n")
    console.print(table)

def check_services():
    install_dir = Path.home() / "llm_system"

    if not install_dir.exists():
        console.print("[red]System not installed![/red]")
        return

    table = Table(title="System Components")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="yellow")

    components = {
        'LLM Engine': 'system/llm_engine.py',
        'Speech Engine': 'system/speech_engine.py',
        'Agent Manager': 'system/agent_manager.py',
        'Coding Assistant': 'system/coding_assistant.py',
        'Distributed Trainer': 'system/distributed_trainer.py',
        'API Server': 'system/api_server.py'
    }

    for name, path in components.items():
        full_path = Path(__file__).parent.parent / path
        status = "[green]✓[/green]" if full_path.exists() else "[red]✗[/red]"
        table.add_row(name, status)

    console.print("\n")
    console.print(table)

def clean_cache():
    cache_dir = Path.home() / "llm_system" / "cache"

    if not cache_dir.exists():
        console.print("[yellow]No cache directory found[/yellow]")
        return

    size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())

    import shutil
    for item in cache_dir.iterdir():
        if item.name.startswith('temp_'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    size_after = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())

    freed = (size_before - size_after) / (1024**2)
    console.print(f"[green]✓ Cleaned cache: freed {freed:.2f} MB[/green]")

def backup_config():
    config_dir = Path.home() / "llm_system" / "configs"
    backup_dir = Path.home() / "llm_system" / "backups"
    backup_dir.mkdir(exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for config_file in config_dir.glob("*.yaml"):
        backup_file = backup_dir / f"{config_file.stem}_{timestamp}.yaml"
        import shutil
        shutil.copy2(config_file, backup_file)
        console.print(f"[green]✓ Backed up {config_file.name} to {backup_file.name}[/green]")

def show_menu():
    menu = """
[bold cyan]System Manager[/bold cyan]

1. Show system status
2. Check installed models
3. Check system components
4. Clean cache
5. Backup configuration
6. View logs
0. Exit
    """
    console.print(Panel(menu, border_style="cyan"))

def view_logs():
    log_dir = Path.home() / "llm_system" / "logs"

    if not log_dir.exists():
        console.print("[yellow]No logs directory found[/yellow]")
        return

    logs = list(log_dir.glob("*.log"))

    if not logs:
        console.print("[yellow]No log files found[/yellow]")
        return

    console.print("\n[cyan]Available logs:[/cyan]")
    for i, log in enumerate(logs, 1):
        console.print(f"{i}. {log.name}")

    choice = console.input("\n[cyan]Select log to view (0 to cancel):[/cyan] ")

    try:
        idx = int(choice)
        if idx == 0:
            return
        if 1 <= idx <= len(logs):
            log_file = logs[idx - 1]
            with open(log_file, 'r') as f:
                content = f.read()
            console.print(Panel(content, title=log_file.name, border_style="yellow"))
    except (ValueError, IndexError):
        console.print("[red]Invalid selection[/red]")

def main():
    console.print("[bold green]Local LLM System Manager[/bold green]\n")

    while True:
        show_menu()

        choice = console.input("\n[bold]Select option:[/bold] ").strip()

        if choice == '0':
            console.print("[yellow]Exiting...[/yellow]")
            break

        elif choice == '1':
            show_system_status()

        elif choice == '2':
            check_models()

        elif choice == '3':
            check_services()

        elif choice == '4':
            clean_cache()

        elif choice == '5':
            backup_config()

        elif choice == '6':
            view_logs()

        else:
            console.print("[red]Invalid option![/red]")

        console.input("\n[dim]Press Enter to continue...[/dim]")

if __name__ == "__main__":
    main()
