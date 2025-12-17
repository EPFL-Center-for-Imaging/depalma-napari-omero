import os
import argparse
from typing import Any, Callable, Dict
import questionary

from depalma_napari_omero.omero_client._project import (
    OmeroController,
    OmeroProjectManager,
)
from mousetumorpy import NNUNET_MODELS, YOLO_MODELS


def cli_menu(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        while True:
            clear_screen()
            out = func(*args, **kwargs)
            if out in ["Back", "back", "🔙 Back"]:
                break

    return wrapper


def clear_screen() -> None:
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def handle_exit(controller: OmeroController):
    print("Bye!")
    controller.quit()
    exit(0)


def handle_login() -> OmeroController:
    """Login to OMERO with username and password"""
    clear_screen()
    max_attempts = 3
    for n_attempts in range(max_attempts):
        user = questionary.text("OMERO username:").ask()
        password = questionary.password("OMERO password:").ask()

        controller = OmeroController(user, password)

        connect_status = controller.connect()
        if connect_status:
            break
        else:
            print(f"{connect_status=}")
            if n_attempts + 1 > max_attempts:
                print(f"Failed to connect {max_attempts} times in a row. Exiting...")
                controller.quit()
                exit(0)

    return controller


@cli_menu
def project_menu(project: OmeroProjectManager) -> str:
    project.scanner.view.print_summary()

    project_choices = {
        "🔙 Back": "back",
        "🔁 Run all workflows": "run_workflows",
        f"🐭 Select cases ({len(project.scanner.view.cases)})": "select_cases",
        "⏫ Import new raw scans in batch": "upload_new_scans",
        f"": "",
    }

    selected_project_option = questionary.select(
        f"What to do next?",
        choices=list(project_choices.keys()),
    ).ask()

    selected_option = project_choices[selected_project_option]

    if selected_option == "run_workflows":
        clear_screen()
        if len(project.scanner.view.roi_missing) or len(project.scanner.view.pred_missing): # type: ignore
            if len(project.scanner.view.roi_missing) != 0: # type: ignore
                lungs_model = questionary.select(
                    "Lungs detection model",
                    choices=project.lungs_models,
                ).ask()
                clear_screen()
            tumor_model = questionary.select(
                "Tumor detection model",
                choices=project.tumor_models,
            ).ask()
            if len(project.scanner.view.roi_missing) != 0: # type: ignore
                project.batch_roi(lungs_model, ask_confirm=False)
            project.batch_nnunet(tumor_model, ask_confirm=False)

        project.batch_track()
        input("\n✅ Press [Enter] to return to the previous menu...")

    elif selected_option == "select_cases":
        select_case_menu(project)

    elif selected_option == "upload_new_scans":
        image_dir = questionary.path(
            "Path to the parent folder containing scan directories",
            only_directories=True,
        ).ask()

        confirm = (
            input("\n✅ Press any key to confirm or [n] to cancel:").strip().lower()
        )

        if confirm == "n":
            print("❌ Cancelled.")
        else:
            for _ in project.upload_from_parent_directory(image_dir):
                pass

            input("\n✅ Press [Enter] to return to the previous menu...")

    return selected_option


@cli_menu
def select_case_menu(project: OmeroProjectManager) -> str:
    choices = ["🔙 Back"] + project.scanner.view.cases

    selected_case = questionary.select(
        "Select a case to work on", choices=choices
    ).ask()

    if selected_case in project.scanner.view.cases:
        case_menu(selected_case, project)

    return selected_case


@cli_menu
def case_menu(selected_case: str, project: OmeroProjectManager) -> str:
    print("\n" + "=" * 60)
    print(f"🐭 Selected case: {selected_case}")

    case_choices = {
        "🔙 Back": "back",
        "⏬ Download case data locally": "download_case",
    }

    selected_case_option = questionary.select(
        f"What to do next?",
        choices=list(case_choices.keys()),
    ).ask()

    selected_option = case_choices[selected_case_option]

    if selected_option == "download_case":
        out_dir = questionary.path(
            "Output path",
            default="questionary",
            only_directories=True,
        ).ask()

        project.download_case(selected_case, out_dir)

    return selected_option


@cli_menu
def interactive(controller: OmeroController) -> str:
    project_choices: Dict[str, Any] = {"🚪 Exit": None}
    for project_name, project_id in controller.projects.items():
        project_choices[f"{project_id} - {project_name}"] = (
            project_id,
            project_name,
        )

    selected_option = questionary.select(
        "Select an OMERO Project to work on",
        choices=list(project_choices.keys()),
    ).ask()

    if selected_option == "🚪 Exit":
        handle_exit(controller)

    selected_project_id, selected_project_name = project_choices[selected_option]

    project = controller.set_project(selected_project_id, selected_project_name, launch_scan=True)

    project_menu(project)

    return selected_option


def run_all_workflows(
    controller: OmeroController, project_id: int, lungs_model: str, tumor_model: str
) -> None:
    """Run all workflows on a given OMERO project"""
    for project_name, omero_project_id in controller.projects.items():
        if project_id == omero_project_id:
            break
    else:
        raise ValueError(
            f"Could not find project with ID {project_id} among available projects: {list(controller.projects.values())}"
        )

    project = controller.set_project(project_id, project_name, launch_scan=True)

    project.scanner.view.print_summary()

    if len(project.scanner.view.roi_missing) or len(project.scanner.view.pred_missing): # type: ignore
        if len(project.scanner.view.roi_missing): # type: ignore
            project.batch_roi(lungs_model, ask_confirm=False)
        project.batch_nnunet(tumor_model, ask_confirm=False)
    project.batch_track()


def main():
    parser = argparse.ArgumentParser(description="OMERO - Mousetumorpy CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("interactive", help="Start the interactive mode")

    run_parser = subparsers.add_parser(
        "run", help="Run all workflows on an OMERO project"
    )

    run_parser.add_argument(
        "project_id",
        help="OMERO Project ID",
        type=int,
    )

    run_parser.add_argument(
        "--lungs-model",
        default="v1",
        choices=list(YOLO_MODELS.keys()),
        help="Lungs model to use",
    )

    run_parser.add_argument(
        "--tumor-model",
        default="oct24",
        choices=list(NNUNET_MODELS.keys()),
        help="Tumor model to use",
    )

    args = parser.parse_args()

    if args.command == "interactive":
        controller = handle_login()
        interactive(controller)
    elif args.command == "run":
        controller = handle_login()
        run_all_workflows(
            controller, args.project_id, args.lungs_model, args.tumor_model
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
