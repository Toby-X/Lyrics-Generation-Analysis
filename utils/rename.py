# %%
import os

def rename_files_recursive(directory):
    # Traverse through directories and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the current file path
            file_path = os.path.join(root, file)
            
            # Extract the directory name
            dir_name = os.path.basename(root)
            
            # Create the new file name with directory name appended
            new_file_name = f"{dir_name}_{file}"
            
            # Rename the file
            new_file_path = os.path.join(root, new_file_name)
            os.rename(file_path, new_file_path)
            
            # Print the old and new file names
            print(f"Renamed: {file_path} -> {new_file_path}")

# Specify the directory path where you want to rename the files recursively
directory_path = os.getcwd().replace("utils", "Figures")

# Call the function to rename files recursively
rename_files_recursive(directory_path)

# %%
