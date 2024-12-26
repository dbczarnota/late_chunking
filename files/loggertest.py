from rich.logging import RichHandler
import logging

# Configure the logger with RichHandler and enable markup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],  # Enable markup
)

logger = logging.getLogger("rich")

# Example usage
logger.debug("[bold cyan]This is a debug message.[/bold cyan]")
logger.info("[bold green]This is an info message.[/bold green]")
logger.warning("[bold yellow]This is a warning message.[/bold yellow]")
logger.error("[bold red]This is an error message.[/bold red]")
logger.disabled = True
logger.critical("[bold magenta]This is a critical message.[/bold magenta]")
