#!/bin/bash

echo "Data:"
printf '%*s\n' "$(tput cols)" '' | tr ' ' '-'
if [ ! -d "data" ]; then
    mkdir "data"
    echo "Created directory 'data'."
fi
cd "data"

# List includes Google drive file IDs and their corresponding names
declare -A files
#files["mmnist_easy"]="1TYVxQBCrK-6NxfKZ_hfgaoHNLCx17IK4"
files["mmnist_medium"]="1KREjVsR62_PxYUVjAUYkpi6uMVS1A77m"

# Download and unzip files if they don't exist
for file_key in "${!files[@]}"
do
    file_id="${files[$file_key]}"
    file_name="${file_key}.zip"

    if [ ! -f "$file_name" ]; then
        echo "Downloading $file_name..."
        gdown "https://drive.google.com/uc?id=$file_id" -O "$file_name"

        if [ -f "$file_name" ]; then
            echo "Download complete for $file_name."
            echo "Unzipping $file_name..."
            unzip "$file_name" -d "."
            echo "Unzip operation complete for $file_name."
            rm "$file_name"
        else
            echo "Failed to download $file_name."
        fi
    else
        echo "$file_name already exists."
    fi
done

printf '%*s\n' "$(tput cols)" '' | tr ' ' '-'
echo


# Setup rye
echo "Rye:"
printf '%*s\n' "$(tput cols)" '' | tr ' ' '-'
if ! command -v rye &> /dev/null; then
    echo "rye could not be found, installing..."
    curl -sSf https://rye-up.com/get | bash
else
    echo "rye is already installed."
fi

source "$HOME/.rye/env"
rye sync
printf '%*s\n' "$(tput cols)" '' | tr ' ' '-'
echo

# Fix torch dependencies
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd ':' -)