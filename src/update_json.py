from utils import * 
# Point the Ultralytics settings to the dataset/results folders of the current fold


# -----------------------------------------------
# Parse the --fold argument from the command line
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold number for cross-validation")
args = parser.parse_args()
fold = args.fold  # This is the fold number passed from your main loop
# -----------------------------------------------



# Repository root: two levels up from this file
found_path = str(Path(__file__).resolve().parent.parent)
print(f"Repository root: {found_path}")

settings_path = Path.home() / ".config" / "Ultralytics" / "settings.json"

with open(settings_path, 'r+') as f:
    config = json.load(f)
    # Update the values
    config["datasets_dir"] = os.path.join(found_path, 'coco_dataset', f"fold{fold}_CV")
    # config["weights_dir"] = os.path.join(found_path, 'weight')
    config["folds_dir"] = os.path.join(found_path, 'results', f"fold{fold}_CV")
    
    # Write the updated config back to the file
    f.seek(0)
    json.dump(config, f, indent=4)
    f.truncate()


########################################################################################
# NOTE: this script expects the Ultralytics settings file to exist
# (it is created automatically the first time `ultralytics` is imported).
# Only `datasets_dir` and `folds_dir` are updated; any other field is left untouched.
########################################################################################
