import os
from main import load_labels

# Python couldn't read labels csv file form original dataset, so I parsed
# the labels file with Java program, that's why It's not included here

labels = load_labels()

root_path = os.path.abspath('')
old_path = os.path.join(root_path, 'old_data', 'test')
new_path = os.path.join(root_path, 'data', 'test')

for key in labels.keys():
    # Dataset contains about 1,8k emojis, but I don't need that much
    if key > 207:
        break

    label = labels[key]

    # Create new directory with corresponding label as a name
    new_dir_path = os.path.join(new_path, label)
    os.mkdir(new_dir_path)

    # Go through all subdirectories (emojis from different vendors)
    for subdir in os.listdir(old_path):

        try:
            # Old file
            file = os.path.join(old_path, subdir, str(key) + '.png')

            # Grab the file, rename it to vendor name and move it to a previously created directory
            new_file = os.path.join(new_dir_path, subdir.lower() + '.png')
            os.rename(file, new_file)

        # Vendor doesn't have the emoji
        except FileNotFoundError:
            continue
