from YoloDatasetsTools import DatasetProcessor
from dataclasses import dataclass
from rich.console import Console
from rich.text import Text
from os import system
from rich.panel import Panel
from rich.table import Table


@dataclass
class BaseConfigs:
    path = ""

def main():
    console = Console()
    system("cls")
    text = Text("Welcome! This is YOLO datasets manager. powerd by ALIREZA and compatibeld with ChatGPT", style="bold magenta")    
    console.print(Panel(text, title="YOLO_Datasets_Manager", style="green"))
    console.print("[italic cyan]this code is for deleting a class or combinig a few datasets and simple agmentation for making more data for detecting datasets and segmentation datasets[/italic cyan]")
    console.print("[bold red]# if you dont know how can you use this code just enter 'help' command[/bold red]")
    return

if __name__ == "__main__":
    main()