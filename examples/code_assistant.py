#!/usr/bin/env python3

import sys
from pathlib import Path
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path.home() / "llm_system"))

from system.coding_assistant import CodingAssistant
from system.llm_engine import LLMEngine

console = Console()

def load_config():
    config_path = Path.home() / "llm_system" / "configs" / "system.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def show_menu():
    menu = """
[bold cyan]Coding Assistant Menu[/bold cyan]

1. Generate code
2. Explain code
3. Debug code
4. Refactor code
5. Add tests
6. Code review
7. Optimize code
8. Convert code (language to language)
9. Generate documentation
0. Exit
    """
    console.print(Panel(menu, border_style="cyan"))

def main():
    config = load_config()

    console.print("[yellow]Initializing Coding Assistant...[/yellow]")

    llm = LLMEngine(config)
    assistant = CodingAssistant(config, llm)

    console.print("[green]✓ Coding Assistant ready![/green]\n")

    while True:
        show_menu()

        choice = console.input("\n[bold]Select option:[/bold] ").strip()

        if choice == '0':
            console.print("[yellow]Exiting...[/yellow]")
            break

        elif choice == '1':
            description = console.input("\n[cyan]Describe what you want to build:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            console.print("\n[yellow]Generating code...[/yellow]\n")

            code = assistant.generate_code(description, language)

            console.print(Panel(code, title=f"Generated {language} Code", border_style="green"))

            save = console.input("\n[cyan]Save to file? (y/n):[/cyan] ").lower()
            if save == 'y':
                filename = console.input("[cyan]Filename:[/cyan] ")
                with open(filename, 'w') as f:
                    f.write(code)
                console.print(f"[green]✓ Saved to {filename}[/green]")

        elif choice == '2':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Analyzing code...[/yellow]\n")

                explanation = assistant.explain_code(code, language)

                md = Markdown(explanation)
                console.print(Panel(md, title="Code Explanation", border_style="blue"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '3':
            filepath = console.input("\n[cyan]Path to buggy code:[/cyan] ")
            error = console.input("[cyan]Error message:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Debugging...[/yellow]\n")

                fix = assistant.debug_code(code, error, language)

                console.print(Panel(fix, title="Debug Solution", border_style="green"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '4':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Refactoring...[/yellow]\n")

                refactored = assistant.refactor_code(code, language)

                console.print(Panel(refactored, title="Refactored Code", border_style="green"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '5':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Generating tests...[/yellow]\n")

                tests = assistant.add_tests(code, language)

                console.print(Panel(tests, title="Generated Tests", border_style="green"))

                save = console.input("\n[cyan]Save tests? (y/n):[/cyan] ").lower()
                if save == 'y':
                    test_file = filepath.replace('.py', '_test.py')
                    with open(test_file, 'w') as f:
                        f.write(tests)
                    console.print(f"[green]✓ Saved to {test_file}[/green]")

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '6':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Performing code review...[/yellow]\n")

                review = assistant.code_review(code, language)

                md = Markdown(review)
                console.print(Panel(md, title="Code Review", border_style="yellow"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '7':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Optimizing code...[/yellow]\n")

                optimized = assistant.optimize_code(code, language)

                console.print(Panel(optimized, title="Optimized Code", border_style="green"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '8':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            from_lang = console.input("[cyan]From language:[/cyan] ")
            to_lang = console.input("[cyan]To language:[/cyan] ")

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print(f"\n[yellow]Converting from {from_lang} to {to_lang}...[/yellow]\n")

                converted = assistant.convert_code(code, from_lang, to_lang)

                console.print(Panel(converted, title=f"Converted {to_lang} Code", border_style="green"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        elif choice == '9':
            filepath = console.input("\n[cyan]Path to code file:[/cyan] ")
            language = console.input("[cyan]Language (default: python):[/cyan] ").strip() or 'python'
            doc_type = console.input("[cyan]Doc type (inline/external):[/cyan] ").strip() or 'inline'

            try:
                with open(filepath, 'r') as f:
                    code = f.read()

                console.print("\n[yellow]Generating documentation...[/yellow]\n")

                docs = assistant.generate_documentation(code, language, doc_type)

                console.print(Panel(docs, title="Documentation", border_style="blue"))

            except FileNotFoundError:
                console.print("[red]File not found![/red]")

        else:
            console.print("[red]Invalid option![/red]")

        console.input("\n[dim]Press Enter to continue...[/dim]")

if __name__ == "__main__":
    main()
