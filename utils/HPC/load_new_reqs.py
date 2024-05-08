import subprocess
import pkg_resources

def install_missing_packages(requirements_file):
    "Only installing packages not already installed. When we have multiple requirements.txt files from the other github sources "
    
    # Read the requirements file and split it into individual package names
    with open(requirements_file, 'r') as file:
        required_packages = file.read().splitlines()

    # Get a list of installed package names
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    # Determine missing packages
    missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

    # Install missing packages
    if missing_packages:
        print("Installing missing packages...")
        for package in missing_packages:
            try:
                subprocess.check_call(['pip', 'install', package])
            except Exception as e:
                print(f"Error installing {package}: {e}")
                pass  # Ignore errors and continue with the next package
        print("Installation complete.")
    else:
        print("All required packages are already installed.")

if __name__ == "__main__":
    requirements_file = '/work3/s174139/Master_Thesis/MagFace-main/raw/requirements.txt'  # Change this to your requirements.txt path - This is from Magface
    install_missing_packages(requirements_file)