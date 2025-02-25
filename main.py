from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from YoloDatasetsTools import DatasetCleaner, DatasetProcessor
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from time import sleep
from pathlib import Path
import shutil
import os
import re

console = Console()

def display_classes(datasets_path, output_path):
    """Display classes in the dataset."""
    path = output_path if output_path != '' else datasets_path[0]
    dataset_path = Prompt.ask("\n> Dataset path", default=path)
    cleaner = DatasetCleaner(dataset_path)
    table = Table(title="Class List", style="magenta")
    table.add_column("ID", style="bold cyan", justify="right")
    table.add_column("Name", style="bold green", justify="left")
    table.add_column("Train", style="bold green", justify="left")
    table.add_column("Valid", style="bold green", justify="left")
    table.add_column("Test", style="bold green", justify="left")
    SUM = 0
    for i, class_name in enumerate(cleaner.classes):
        train = cleaner.count_class_samples(class_name=class_name, subset= 'train')
        valid = cleaner.count_class_samples(class_name=class_name, subset= 'valid')
        test = cleaner.count_class_samples(class_name=class_name, subset= 'test')
        table.add_row(str(i), class_name, str(train), str(valid), str(test))
        # SUM
    console.print(table)

def count_samples(datasets_path, output_path):
    """Count samples for a specific class."""
    path = output_path if output_path != '' else datasets_path[0]
    dataset_path = Prompt.ask("Dataset path", default=path)
    cleaner = DatasetCleaner(dataset_path)
    class_name = Prompt.ask("Enter the class name to count samples for")
    subset = Prompt.ask("Enter subset (train, valid, test) or leave blank for all", default=None)
    try:
        count = cleaner.count_class_samples(class_name=class_name, subset=subset)
        console.print(f"[bold green]Found {count} samples for class '{class_name}' in subset(s): {subset or 'all'}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

def delete_classes(datasets_path, output_path):
    """Delete one or more classes from the dataset."""
    path = output_path if output_path != '' else datasets_path[0]
    dataset_path = Prompt.ask("Dataset path", default=path)
    cleaner = DatasetCleaner(dataset_path)
    class_names = Prompt.ask("Enter class names to delete (comma-separated)").split(",")
    subset = Prompt.ask("Enter subset (train, valid, test) or leave blank for all", default=None)
    max_samples = Prompt.ask("Enter max samples to delete per class (or leave blank for all)", default=None)
    max_samples = int(max_samples) if max_samples else None

    confirm = Prompt.ask("[bold yellow]Are you sure you want to delete these classes? (yes/no)[/bold yellow]", choices=["yes", "no"], default="no")
    if confirm == "no":
        console.print("[bold yellow]Operation cancelled.[/bold yellow]")
        return

    with Progress() as progress:
        task = progress.add_task("[red]Deleting classes...", total=len(class_names))
        for class_name in class_names:
            cleaner.delete_class(class_names=[class_name.strip()], max_samples=max_samples, subset=subset)
            progress.update(task, advance=1)

    console.print("[bold green]Classes deleted successfully![/bold green]")

def augment_dataset(processor, in_path='', out_path=''):
    """Augment dataset using DatasetProcessor."""
    dataset_path = Prompt.ask("Enter the dataset path for augmentation", default=in_path)
    output_path = Prompt.ask("Enter the output path for augmented dataset", default=out_path)
    doyou = Prompt.ask("Do you want to change augmentation prametrs (Y/N)? ", default="N").upper()
    doyou = False if doyou == 'Y' else True
    augmen_params, multiplier = augmentation_params(doyou)
    os.system("cls")
    processor.process_folder(dataset_path, output_path, augmen_params, multiplier)
    
def resize_and_crop(processor, in_paths, out_path=''):
    size = (640, 640)
    in_path = in_paths[0]
    if len(in_paths) >= 2:
        in_path = out_path
    dataset_path = Prompt.ask("Enter the dataset path", default=in_path)
    output_path = Prompt.ask("Enter the output path", default=out_path)
    write_typly(
'''>[0]Compression: Resize with compressing images
>[1]Advanced_compression: Resize with advanced compressing images
>[2]Crop: Resize with cropping images
>[3]Advanced_crop: Resize with advanced cropping images\n'''
)
    mode = int(Prompt.ask("Choose a number for resize mode of images", default=f"0").strip())
    match mode:
        case 0:
            mode = "fixed_resize"
        case 1:
            mode = "advance_resize"
        case 2:
            mode = "fixed_crop"
        case 3:
            mode = "advance_crop"
        case _:
            mode = "fixed_resize"
    _crop = None
    if mode == "fixed_crop":
        _crop = Prompt.ask("Target crop size (Xmin, Ymin, Xmax, Ymax) or offset number", default=f"10")
        _crop = re.findall(r"\d+", _crop)
        _crop = list(map(lambda x:int(x), _crop))
        if len(_crop) == 1:
            _crop = _crop[0]
    else:
        sizeprompt = Prompt.ask("Enter the output size of images", default=f"{size[0]}, {size[1]}")
        sizeprompt = re.findall(r"\d+", sizeprompt)
        if len(sizeprompt) == 1:
            size = (int(sizeprompt[0]), int(sizeprompt[0]))
        elif len(sizeprompt) >= 2:
            size = (int(sizeprompt[0]), int(sizeprompt[1]))
        write_typly(f"Change size configed to {size}\n")
    processor.process_resize_and_crop(dataset_path, output_path, size, mode=mode, fixed_crop=_crop)

def combine_datasets(combiner, paths, out_path=''):
    """Combine multiple YOLO datasets."""
    dataset_paths = Prompt.ask("Enter the paths to datasets to combine (comma-separated)").split(",")
    dataset_paths = [path.strip() for path in dataset_paths]
    if dataset_paths == ['']:
        dataset_paths = paths
    write_typly(f"> Combining on {dataset_paths}\n")
    output_path = Prompt.ask("Enter the output path for combined dataset", default=out_path)
    write_typly(f"> Output Dataset in ({output_path}) folder\n")
    doyou = Prompt.ask("Do you want to have augmentation on your datasets (Y/N)? ", default="Y").upper()
    doyou = True if doyou == 'Y' else False
    if doyou:
        adoyou = Prompt.ask("Do you want to change augmentation prametrs (Y/N)? ", default="N").upper()
        adoyou = False if doyou == 'Y' else True
        augmen_params, multiplier = augmentation_params(adoyou)
    os.system("cls")
    combiner.combine_datasets(dataset_paths, output_path)
    if doyou:
        combiner.process_folder(output_path, output_path, augmen_params, multiplier)

def equalization(datasets_path, output_path):
    path = output_path if output_path != '' else datasets_path[0]
    cleaner = DatasetCleaner(path)
    dataset_path = Prompt.ask("Dataset path", default=path)
    subset = Prompt.ask("Enter subset (train, valid, test) or leave blank for all", default=None)
    cleaner.classes_equalization(subset=subset)
    print("Classes equalized successfully!")

def visualize(visual, dataset_path, out_path=''):
    path = dataset_path if out_path == '' else out_path
    path = Prompt.ask("Enter the dataset path for augmentation", default=path)
    PATH = Path() / path / 'visualized'
    if os.path.exists(PATH): 
        last_check_point = True if Prompt.ask("Do you want to continue last check point(Y/N)", default="Y").upper().strip() == 'Y' else False
        if not last_check_point:
            shutil.rmtree(PATH)
    check_mode = True if Prompt.ask("Do you want to check the datasets class (Y/N)", default="Y").upper().strip() == 'Y' else False
    folders = Prompt.ask("Enter the paths to datasets to combine (comma-separated)", default="train").split(",")
    folders = [folder.strip() for folder in folders]
    if folders == ['']:
        folders = None
    visual.visualize_annotations(path, check=check_mode, folders=folders)

def shuffle(processor, datasets_path, output_path):
    path = output_path if output_path != '' else datasets_path[0]
    dataset_paths = Prompt.ask("Enter the paths to datasets to c (comma-separated)", default=path).split(",")
    dataset_paths = [path.strip() for path in dataset_paths]
    if dataset_paths == ['']:
        dataset_paths = [path]
    for dataset in dataset_paths:
        print(f"Process: {dataset}")
        processor.shuffle_and_rename_dataset(dataset)

def Segmentation_to_detection(processor, datasets_path, output_path=''):
    path = output_path if output_path != '' else datasets_path[0]
    dataset_paths = Prompt.ask("Enter the paths to datasets to c (comma-separated)", default=path).split(",")
    dataset_paths = [path.strip() for path in dataset_paths]
    if dataset_paths == ['']:
        dataset_paths = [path]
    for dataset in dataset_paths:
        print(f"Process: {dataset}")
        processor.segmentation_to_detection(dataset)
        
def display_menu(processor, datasets_path, output_path):
    """Main menu for dataset management."""
    while True:
        console.print(Panel("[bold cyan]YOLO Dataset Management Tool[/bold cyan]", style="bold magenta"))
        if len(datasets_path) >= 2:
            console.print(
                "[1] Display classes\n"
                "[2] Count samples for a class\n"
                "[3] Delete a class (or multiple classes)\n"
                "[4] Augment dataset\n"
                "[5] visualize_annotations_bounding_boxs\n"
                "[6] Resize\n"
                "[7] Class equalization\n"
                "[8] Segmentation to detection\n"
                "[9] shuffle and rename dataset\n"
                "[10] Combine datasets\n"
                "[11] Exit"
            )
            choice = Prompt.ask("Choose an option (1-7)")
        else:
            console.print(
                "[1] Display classes\n"
                "[2] Count samples for a class\n"
                "[3] Delete a class (or multiple classes)\n"
                "[4] Augment dataset\n"
                "[5] visualize_annotations_bounding_boxs\n"
                "[6] Resize\n"
                "[7] Class equalization\n"
                "[8] Segmentation to detection\n"
                "[9] shuffle and rename dataset\n"
                "[10] Exit"
            )
            choice = Prompt.ask("Choose an option (1-6)")
        
        if choice == "1":
            display_classes(datasets_path, output_path)
        elif choice == "2":
            count_samples(datasets_path, output_path)
        elif choice == "3":
            delete_classes(datasets_path, output_path)
        elif choice == "4":
            augment_dataset(processor, datasets_path[0], output_path)
        elif choice == "5":
            visualize(processor, datasets_path[0], output_path)
        elif choice == "6":
            resize_and_crop(processor, datasets_path, output_path)
        elif choice == "7":
            equalization(datasets_path, output_path)
        elif choice == "8":
            Segmentation_to_detection(processor, datasets_path, output_path)
        elif choice == "9":
            shuffle(processor, datasets_path, output_path)
        elif choice == "10" and len(datasets_path) >= 2:
            combine_datasets(processor, datasets_path, output_path)
        elif choice == "11" and len(datasets_path) >= 2:
            console.print("[bold yellow]Exiting the tool. Goodbye![/bold yellow]")
            break
        elif choice == "10" and not len(datasets_path) >= 2:
            console.print("[bold yellow]Exiting the tool. Goodbye![/bold yellow]")
            break
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")

def create_style():
    return Style.from_dict({
        '': '#ffffff',
        'path': 'ansicyan bold', 
        'input': 'ansigreen', 
        'message': 'ansiyellow',
    })

def custom_prompt(ask, help_ = ''):
    kb = KeyBindings()
    @kb.add('c-c')
    def exit_(event):
        print_formatted_text('Exiting...', style='ansired')
        exit(0)
    ask_list = [help_ , ask]
    completer = PathCompleter(only_directories=True)
    user_input = {}
    while True:
        os.system("cls")
        _input = (prompt(
            " \n".join(str(x) for x in ask_list),
            completer=completer,
            key_bindings=kb,
            style=create_style(), 
            include_default_pygments_style=False
        )).strip()
        show_string = ""
        path_status = False
        if _input == '':
            break
        elif _input == '-1':
            if len(ask_list) <= 2:
                continue
            user_input.pop(ask_list[-1])
            ask_list = ask_list[:-1]
            continue
        elif _input == '-2':
            if len(ask_list) <= 2:
                continue
            user_input.clear()
            ask_list = ask_list[:2]
            continue
        elif _input == '-3':
            if len(ask_list) <= 2:
                continue
            kk = tuple(user_input.items())
            for key, item in kk:
                if not item:
                    user_input.pop(key)
            pp = ask_list[2:]
            ask_list = ask_list[:2]
            pp = list(filter(lambda x:x[-3:] != "-),", pp))
            ask_list += pp
            continue
        
        path = Path() / _input
        if not path.exists():
            show_string = f'(-Not found-> {_input} -),'
        elif path.is_file():
            show_string = f'(-Is a file-> {_input} -),'
        elif path.is_dir() and not (path / 'data.yaml').exists():
            show_string = f'(-Not Yolo Dataset-> {_input} -),'
        else:
            show_string = f'({_input}),'
            path_status = True
        ask_list.append(show_string) 
        user_input[show_string] = path_status
    path_out_list = []
    for key, item in user_input.items():
        if item:
            path_out_list.append(str(key)[1:-2])
    
    return path_out_list

def write_typly(text, num=0.01): 
    for char in text:
        console.print(char, end="", style="bold green")
        sleep(num)

def get_range_input(param, default_min="0", default_max="1"):
    """Get min and max values for each parameter."""
    min_val = float(prompt(f"Enter minimum value for {param}: ", default=str(default_min), style=Style.from_dict({'': 'fg:yellow'})))
    max_val = float(prompt(f"Enter maximum value for {param}: ", default=str(default_max), style=Style.from_dict({'': 'fg:yellow'})))
    if min_val <= max_val:
        return min_val, max_val
    return default_min, default_max

def get_multiplier(default="1"):
    """Get multiplier value from user."""
    return int(prompt("Enter multiplier value: ", default=str(default), style=Style.from_dict({'': 'fg:yellow'})))

def display_params(augmentation_params, multiplier):
    """Display the final parameters."""
    console = Console()
    console.print("\n[bold green]Updated Augmentation Parameters:[/bold green]")
    console.print(f"{augmentation_params}, multiplier={multiplier}")

def augmentation_params(default=False):
    param_names = {
    'hue': (-10, 10),
    'saturation': (0.7, 1.3),
    'brightness': (0.7, 1.3),
    'contrast': (0.8, 1.2),
    'noise': (10, 50),
    'color_jitter': (0.9, 1.1)
    }
    multiplier = 4
    if default:
        return param_names, multiplier
    augmentation_params = {}
    for param, (_min, _max) in param_names.items():
        console.print(f"[cyan]{param}[/cyan]:", style="bold")
        augmentation_params[param] = get_range_input(param, _min, _max)
    multiplier = get_multiplier(multiplier)
    display_params(augmentation_params, multiplier)
    return augmentation_params, multiplier

def main():
    datasets_paths = custom_prompt("Enter the path to your YOLO dataset: ",
                          f"\n{'-'*30}\n> Empty to continue next\n> Ctrl+C to Exit\n> -1 to Edit\n> -2 to clear all dir paths\n> -3 to clear all wrong paths\n{'-'*30}\n\n")
    write_typly(f"\n{'-'*19}\n- Made By ALIREZA -\n{'-'*19}\n", 0.01)
    output_path = ''
    if len(datasets_paths) > 1:
        output_path = input("\nEnter an output path (defult: combined_dataset): ").strip()
        if output_path == '':
            output_path = "combined_dataset"
    write_typly(f"\n> Datasets path {datasets_paths}\n")
    if output_path != '':
        write_typly(f"> Output path [{output_path}]\n")
    processor = DatasetProcessor(output_path=output_path)
    for dataset_path in datasets_paths:
        if not dataset_path or not os.path.exists(dataset_path):
            console.print("[bold red]Error: Dataset path does not exist.[/bold red]")
            return
        processor.ensure_dataset(dataset_path)

    display_menu(processor, datasets_paths, output_path)

if __name__ == "__main__":
    main()
