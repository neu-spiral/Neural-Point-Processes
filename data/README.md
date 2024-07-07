# Data Folder

This directory is designated for storing the dataset(s) used by this project. To ensure the project functions correctly, please make sure all required data files are within this folder.

## Generating the Data

Instructions for generating the necessary data can be found in the "Data Description and Generation" section of the [README](../README.md) file located in the parent directory. Please follow the guidelines provided there to acquire and organize the data appropriately in this folder.

## Structure

Once you have generated the data, your data directory structure should resemble the following (this is an example; the actual structure and directory names will depend on the data and instructions provided in the "Data Description and Generation" section):

```
data/
│
├── Building
├── Synthetic
└── PinMNIST/
    ├── random_100pins/pins.csv
    ├── random_10pins/pins.csv
    ├── mesh_3step_pins/pins.csv
    ├── mesh_10step_pins/pins.csv
    └── images/
        ├── 0.png
        ├── 1.png
        └──...
```

## Note

If there are any specific requirements or instructions regarding data preprocessing or organization, they will be detailed in the main [README](../README.md)'s "Data Description" section. Ensure to review those details to properly prepare the data for use with this project.