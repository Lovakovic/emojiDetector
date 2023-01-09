import os


def load_labels() -> dict[int, str]:
    """
    Loads labels from a csv file in data dir
    :return: Labels dict containing image name (int) and its name
    """
    labels = {}

    with open(os.path.join('data', 'labels.csv'), 'r') as labels_file:
        lines = labels_file.readlines()

        for line in lines:
            tokens = line.split(',')
            labels[int(tokens[0])] = tokens[1].replace('\n', '')

    return labels


def parse_data():
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


def png_to_jpg():
    root_path = os.path.join(os.path.abspath(''), 'data')

    for dir_name, sub_dir_list, file_list in os.walk(root_path):
        for fname in file_list:
            if fname.endswith('.png'):
                old_path = os.path.join(dir_name, fname)

                new_path = old_path[:-4] + '.jpeg'

                os.rename(old_path, new_path)


png_to_jpg()