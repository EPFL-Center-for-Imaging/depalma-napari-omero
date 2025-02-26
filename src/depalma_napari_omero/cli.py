import os
import argparse
import questionary
import skimage.io
from pathlib import Path

from mousetumorpy import YOLO_MODELS, NNUNET_MODELS

LUNGS_MODELS = list(YOLO_MODELS.keys())
TUMOR_MODELS = list(NNUNET_MODELS.keys())

from mousetumorpy import LungsPredictor, TumorPredictor, combine_images, run_tracking
from depalma_napari_omero.omero_server import OmeroServer
from depalma_napari_omero.project import ProjectRepresentation


def clear_screen():
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def handle_exit(server):
    print("Bye!")
    server.quit()
    exit(0)


def handle_login(server):
    clear_screen()
    # Login to OMERO with username and password
    max_attempts = 3
    for n_attempts in range(max_attempts):
        user = questionary.text("OMERO username:").ask()
        password = questionary.password("OMERO password:").ask()

        server.login(user, password)

        connect_status = server.connect()

        if connect_status:
            break
        else:
            if n_attempts + 1 > max_attempts:
                print(f"Failed to connect {max_attempts} times in a row. Exiting...")
                server.quit()
                exit(0)

    return server


def project_menu(server, project: ProjectRepresentation):
    while True:
        clear_screen()

        project.print_summary()

        project_choices = {
            "Back": "back",
            f"Update ROIs ({len(project.roi_missing)})": "update_rois",
            f"Update Tumor predictions ({len(project.pred_missing)})": "update_preds",
            f"Update ROIs ({len(project.roi_missing)}) and Tumor predictions ({len(project.pred_missing)})": "update_rois_and_preds",
            f"Select cases ({len(project.cases)})": "select_cases",
        }

        selected_project_option = questionary.select(
            f"What to do next?",
            choices=list(project_choices.keys()),
        ).ask()

        choice_made = project_choices.get(selected_project_option)

        if choice_made == "back":
            break

        elif choice_made == "update_rois":
            clear_screen()
            lungs_model = questionary.select(
                "Lungs detection model", choices=LUNGS_MODELS
            ).ask()
            project.batch_roi(lungs_model)
            input("\nPress Enter to return to the previous menu...")

        elif choice_made == "update_preds":
            clear_screen()
            tumor_model = questionary.select(
                "Tumor detection model", choices=TUMOR_MODELS
            ).ask()
            project.batch_nnunet(tumor_model)
            input("\nPress Enter to return to the previous menu...")

        elif choice_made == "update_rois_and_preds":
            clear_screen()
            lungs_model = questionary.select(
                "Lungs detection model", choices=LUNGS_MODELS
            ).ask()
            clear_screen()
            tumor_model = questionary.select(
                "Tumor detection model", choices=TUMOR_MODELS
            ).ask()
            project.batch_roi(lungs_model)
            project.batch_nnunet(tumor_model)
            input("\nPress Enter to return to the previous menu...")

        elif choice_made == "select_cases":
            while True:
                clear_screen()
                choices = ["Back"] + project.cases
                selected_case = questionary.select(
                    "Select a case to work on", choices=choices
                ).ask()
                if selected_case == "Back":
                    break
                else:
                    case_menu(server, selected_case, project)


def case_menu(server, selected_case, project):
    while True:
        clear_screen()

        print(f"Selected case: {selected_case}")

        case_choices = {
            "Back": "back",
            # "Track case": "track_case",
            # "Download case time series": "download_timeseries",
            "Run full pipeline on case": "full_pipeline",
        }

        selected_case_option = questionary.select(
            f"What to do next?",
            choices=list(case_choices.keys()),
        ).ask()

        choice_made = case_choices.get(selected_case_option)

        if choice_made == "back":
            break
        # elif choice_made == "track_case":
        #     print("Tracking...")
        #     input("\nPress Enter to return to the previous menu...")
        # elif choice_made == "download_timeseries":
        #     print("Downloading...")
        #     input("\nPress Enter to return to the previous menu...")
        elif choice_made == "full_pipeline":
            # Output folder?
            # ...
            lungs_model = questionary.select(
                "Lungs detection model", choices=LUNGS_MODELS
            ).ask()
            tumor_model = questionary.select(
                "Tumor detection model", choices=TUMOR_MODELS
            ).ask()
            out_folder = questionary.path(
                "Output path",
                default="questionary",
                only_directories=True,
            ).ask()

            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
                print("Created the output folder: ", out_folder)
            
            out_folder = Path(out_folder)

            to_download_image_timeseries_ids = project.image_timeseries_ids(
                selected_case
            )

            images = [
                server.download_image(img_id)
                for img_id in to_download_image_timeseries_ids
            ]

            for k, image in enumerate(images):
                out_file = str(out_folder / f"SCAN{k:02d}.tif")
                skimage.io.imsave(out_file, image)

            # clear_screen()

            predictor = LungsPredictor(lungs_model)

            rois = []
            lungs_rois = []
            for k, image in enumerate(images):
                roi, lungs_roi = predictor.compute_3d_roi(image)
                rois.append(roi)
                lungs_rois.append(lungs_roi)

            rois_timeseries = combine_images(rois)
            skimage.io.imsave(str(out_folder / "rois_timeseries.tif"), rois_timeseries)

            lungs_timeseries = combine_images(lungs_rois)
            skimage.io.imsave(
                str(out_folder / "lungs_timeseries.tif"), lungs_timeseries
            )

            # clear_screen()

            predictor = TumorPredictor(tumor_model)

            tumors_rois = []
            for k, image in enumerate(images):
                tumor_mask = predictor.predict(image)
                tumors_rois.append(tumor_mask)

            tumor_timeseries = combine_images(tumors_rois)
            skimage.io.imsave(
                str(out_folder / "tumor_timeseries.tif"), tumor_timeseries
            )

            # Align before tracking?
            # ...

            pivoted_df, tumor_timeseries_corrected = run_tracking(
                tumor_timeseries,
                rois_timeseries,
                lungs_timeseries,
                with_lungs_registration=True,  # Optional?
                method="laptrack",
                max_dist_px=30,
                dist_weight_ratio=0.9,
                max_volume_diff_rel=1.0,
                memory=0,
            )

            skimage.io.imsave(
                str(out_folder / "tumor_timeseries_corrected.tif"),
                tumor_timeseries_corrected,
            )

            pivoted_df.to_csv(str(out_folder / f"{selected_case}_results.csv"))


def interactive(server: OmeroServer):
    while True:
        clear_screen()

        project_choices = {"Exit": None}
        for project_name, project_id in server.projects.items():
            project_choices[f"{project_id} - {project_name}"] = (
                project_id,
                project_name,
            )

        selected_option = questionary.select(
            "Select an OMERO Project to work on",
            choices=list(project_choices.keys()),
        ).ask()
        if selected_option == "Exit":
            handle_exit(server)
        else:
            selected_project_id, selected_project_name = project_choices[
                selected_option
            ]
            project = ProjectRepresentation(
                server, selected_project_id, selected_project_name
            )
            project_menu(server, project)


def main():
    parser = argparse.ArgumentParser(description="OMERO - Mousetumorpy CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("interactive", help="Start the interactive mode")

    args = parser.parse_args()

    if args.command == "interactive":
        server = handle_login(OmeroServer())
        interactive(server)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
